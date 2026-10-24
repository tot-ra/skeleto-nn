// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Artjom Kurapov
// One private child per visible hero. No sockets, Python, or game-state writes.
#include <motionbricks/inference.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr unsigned joints = 34;
constexpr unsigned stride = joints * 4;
char error_buffer[2048]{};

void check(mb_status status) {
    if (status != MB_OK) throw std::runtime_error(error_buffer);
}
std::string quoted(const std::string & value) {
    std::string result = "\"";
    for (unsigned char ch : value) {
        if (ch == '"' || ch == '\\') result += '\\';
        if (ch < 32) result += ' ';
        else result += static_cast<char>(ch);
    }
    return result + '"';
}
template<class T> void array(std::ostream & out, const T * data, std::size_t count) {
    out << '[';
    for (std::size_t i = 0; i < count; ++i) {
        if (i) out << ',';
        out << data[i];
    }
    out << ']';
}

struct Runtime {
    mb_runtime_options * options{};
    mb_model * model{};
    mb_agent * agent{};
    mb_command * command{};
    mb_motion * motion{};
    std::map<std::string, mb_style *> styles;
    std::map<std::string, unsigned> indices;
    std::vector<float> last_roots, last_rotations;
    uint64_t last_frames{};

    ~Runtime() {
        mb_motion_free(motion);
        mb_command_free(command);
        mb_agent_free(agent);
        for (auto & pair : styles) mb_style_free(pair.second);
        mb_model_free(model);
        mb_runtime_options_free(options);
    }

    void load(const std::string & directory, unsigned threads) {
        check(mb_runtime_options_create(&options, error_buffer, sizeof error_buffer));
        check(mb_runtime_options_set_device(options, MB_DEVICE_CPU, error_buffer, sizeof error_buffer));
        check(mb_runtime_options_set_threads(options, threads, error_buffer, sizeof error_buffer));
        check(mb_model_load((directory + "/g1-f32").c_str(), options, &model, error_buffer, sizeof error_buffer));
        for (const std::string name : {"idle", "walk", "slow_walk"}) {
            mb_style * value{};
            check(mb_style_load(model, (directory + "/styles/" + name + ".mbstyle").c_str(),
                                &value, error_buffer, sizeof error_buffer));
            styles[name] = value;
        }
        check(mb_agent_create(model, &agent, error_buffer, sizeof error_buffer));
        check(mb_agent_reset(agent, styles.at("idle"), error_buffer, sizeof error_buffer));
        check(mb_command_create(&command, error_buffer, sizeof error_buffer));
        std::cout << "{\"protocol\":\"skeleto-nn/motion/1\",\"fps\":30,\"threads\":"
                  << threads << ",\"joints\":[";
        for (unsigned i = 0; i < joints; ++i) {
            const char * name{};
            int32_t parent{};
            float x{}, y{}, z{};
            check(mb_model_get_joint_name(model, i, &name, error_buffer, sizeof error_buffer));
            check(mb_model_get_joint_parent(model, i, &parent, error_buffer, sizeof error_buffer));
            check(mb_model_get_neutral_joint_position(model, i, &x, &y, &z, error_buffer, sizeof error_buffer));
            indices[name] = i;
            if (i) std::cout << ',';
            std::cout << "{\"name\":" << quoted(name) << ",\"parent\":" << parent
                      << ",\"position\":[" << x << ',' << y << ',' << z << "]}";
        }
        std::cout << "]}" << std::endl;
    }

    void context(unsigned advance) {
        if (!last_frames) return;
        const auto first = std::min<uint64_t>(advance, last_frames - 4);
        check(mb_agent_set_context(agent, last_roots.data() + first * 3,
            last_rotations.data() + first * stride, 4, joints, error_buffer, sizeof error_buffer));
    }

    // Goal keyframes constrain the pretrained in-betweening model. These are
    // not output clips: mb_model_infer generates the transition in every call.
    void rotate_goal(std::vector<float> & rotations, unsigned frame,
                     const std::string & name, unsigned axis, float angle) {
        const auto found = indices.find(name);
        if (found == indices.end()) throw std::runtime_error("goal joint missing: " + name);
        float * q = rotations.data() + frame * stride + found->second * 4;
        float v[3]{};
        v[axis] = std::sin(angle * 0.5F);
        const float w = std::cos(angle * 0.5F);
        const std::array<float, 4> result{
            w*q[0]+v[0]*q[3]+v[1]*q[2]-v[2]*q[1],
            w*q[1]-v[0]*q[2]+v[1]*q[3]+v[2]*q[0],
            w*q[2]+v[0]*q[1]-v[1]*q[0]+v[2]*q[3],
            w*q[3]-v[0]*q[0]-v[1]*q[1]-v[2]*q[2]};
        std::copy(result.begin(), result.end(), q);
    }

    void constrained_goal(const std::string & mode, float phase, uint64_t seed) {
        uint64_t frames{}, count{};
        const float * roots{};
        const float * rotations{};
        check(mb_motion_get_frame_count(motion, &frames, error_buffer, sizeof error_buffer));
        check(mb_motion_get_root_translations(motion, &roots, &count, error_buffer, sizeof error_buffer));
        std::vector<float> source_roots(roots, roots + 12);
        check(mb_motion_get_local_rotations_xyzw(motion, &rotations, &count, error_buffer, sizeof error_buffer));
        std::vector<float> source_rotations(rotations, rotations + 4 * stride);
        auto target_roots = source_roots;
        auto target_rotations = source_rotations;
        for (unsigned f = 0; f < 4; ++f) {
            if (mode == "door") {
                rotate_goal(target_rotations, f, "right_shoulder_pitch_skel", 0, -0.9F);
                rotate_goal(target_rotations, f, "right_elbow_skel", 0, -0.45F);
            } else if (mode == "ladder") {
                for (const std::string side : {"left", "right"}) {
                    const float cycle = std::sin(6.2831853F * phase + (side == "left" ? 0.0F : 3.1415927F));
                    rotate_goal(target_rotations, f, side + "_shoulder_pitch_skel", 0, -1.7F - cycle * 0.3F);
                    rotate_goal(target_rotations, f, side + "_elbow_skel", 0, -0.7F);
                    rotate_goal(target_rotations, f, side + "_hip_pitch_skel", 0, -0.45F - cycle * 0.3F);
                    rotate_goal(target_rotations, f, side + "_knee_skel", 0, 0.85F + cycle * 0.45F);
                }
            } else {
                const float bend = mode == "jump_air" ? 0.85F :
                    (mode == "jump_land" ? 0.55F * (1.0F-phase) : 0.6F);
                for (const std::string side : {"left", "right"}) {
                    rotate_goal(target_rotations, f, side + "_hip_pitch_skel", 0, -bend * 0.5F);
                    rotate_goal(target_rotations, f, side + "_knee_skel", 0, bend);
                    rotate_goal(target_rotations, f, side + "_ankle_pitch_skel", 0, -bend * 0.5F);
                    rotate_goal(target_rotations, f, side + "_shoulder_pitch_skel", 0, -0.35F);
                }
            }
        }
        mb_inference_request * request{};
        mb_motion * generated{};
        check(mb_inference_request_create(&request, error_buffer, sizeof error_buffer));
        try {
            check(mb_inference_request_set_boundary_poses(request, model, 0,
                source_roots.data(), source_roots.size(), source_rotations.data(),
                source_rotations.size(), error_buffer, sizeof error_buffer));
            check(mb_inference_request_set_boundary_poses(request, model, 1,
                target_roots.data(), target_roots.size(), target_rotations.data(),
                target_rotations.size(), error_buffer, sizeof error_buffer));
            const uint32_t durations[11]{1,0,0,0,0,0,0,0,0,0,0};
            check(mb_inference_request_set_mask(request, MB_INFERENCE_DURATIONS,
                durations, 11, error_buffer, sizeof error_buffer));
            check(mb_inference_request_set_seed(request, seed, error_buffer, sizeof error_buffer));
            check(mb_model_infer(model, request, &generated, error_buffer, sizeof error_buffer));
        } catch (...) {
            mb_motion_free(generated);
            mb_inference_request_free(request);
            throw;
        }
        mb_inference_request_free(request);
        mb_motion_free(motion);
        motion = generated;
    }

    void plan(uint64_t id, unsigned advance, float speed, float mx, float mz,
              float fx, float fz, const std::string & mode, float phase) {
        if (advance > 64 || speed < 0 || speed > 10 || phase < 0 || phase > 1 ||
            !std::isfinite(speed+mx+mz+fx+fz+phase) ||
            std::hypot(fx, fz) < 0.001F || std::hypot(mx, mz) < 0.001F)
            throw std::runtime_error("invalid movement request");
        if (mode != "locomotion" && mode != "jump_start" && mode != "jump_air" &&
            mode != "jump_land" && mode != "door" && mode != "ladder") throw std::runtime_error("unknown motion goal");
        context(advance);
        mb_motion_free(motion);
        motion = nullptr;
        const auto started = std::chrono::steady_clock::now();
        const auto style = styles.at(speed < 0.05F ? "idle" : "walk");
        check(mb_command_set_style(command, style, error_buffer, sizeof error_buffer));
        check(mb_command_set_target_speed(command, speed, error_buffer, sizeof error_buffer));
        check(mb_command_set_movement_direction(command, mx, 0, mz, error_buffer, sizeof error_buffer));
        check(mb_command_set_facing_direction(command, fx, 0, fz, error_buffer, sizeof error_buffer));
        check(mb_command_set_seed(command, id, error_buffer, sizeof error_buffer));
        check(mb_agent_plan(agent, command, &motion, error_buffer, sizeof error_buffer));
        if (mode != "locomotion") constrained_goal(mode, phase, id);
        uint64_t count{};
        const float * roots{};
        const float * rotations{};
        check(mb_motion_get_frame_count(motion, &last_frames, error_buffer, sizeof error_buffer));
        check(mb_motion_get_root_translations(motion, &roots, &count, error_buffer, sizeof error_buffer));
        last_roots.assign(roots, roots + count);
        check(mb_motion_get_local_rotations_xyzw(motion, &rotations, &count, error_buffer, sizeof error_buffer));
        last_rotations.assign(rotations, rotations + count);
        for (float value : last_roots) if (!std::isfinite(value)) throw std::runtime_error("non-finite root");
        for (float value : last_rotations) if (!std::isfinite(value)) throw std::runtime_error("non-finite pose");
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        std::cout << "{\"id\":" << id << ",\"mode\":" << quoted(mode)
                  << ",\"frames\":" << last_frames << ",\"inference_ms\":" << ms << ",\"roots\":";
        array(std::cout, last_roots.data(), last_roots.size());
        std::cout << ",\"rotations\":";
        array(std::cout, last_rotations.data(), last_rotations.size());
        std::cout << '}' << std::endl;
    }
};
}

int main(int argc, char ** argv) {
    std::cout << std::setprecision(9);
    try {
        if (argc != 3) throw std::runtime_error("usage: skeleto_motion_worker BUNDLE_ROOT THREADS");
        const unsigned threads = static_cast<unsigned>(std::stoul(argv[2]));
        if (threads < 1 || threads > 8) throw std::runtime_error("threads must be 1..8");
        Runtime runtime;
        runtime.load(argv[1], threads);
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line == "quit") break;
            try {
                if (line.size() > 4096) throw std::runtime_error("request too large");
                std::istringstream input(line);
                std::string operation, mode, extra;
                uint64_t id{};
                unsigned advance{};
                float speed{}, mx{}, mz{}, fx{}, fz{}, phase{};
                if (!(input >> operation >> id >> advance >> speed >> mx >> mz >> fx >> fz >> mode >> phase)
                    || operation != "plan" || (input >> extra)) throw std::runtime_error("malformed request");
                runtime.plan(id, advance, speed, mx, mz, fx, fz, mode, phase);
            } catch (const std::exception & error) {
                std::cout << "{\"error\":" << quoted(error.what()) << '}' << std::endl;
            }
        }
        return 0;
    } catch (const std::exception & error) {
        std::cout << "{\"error\":" << quoted(error.what()) << '}' << std::endl;
        return 2;
    }
}
