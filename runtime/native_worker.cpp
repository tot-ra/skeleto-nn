// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Artjom Kurapov
// One private child per visible hero. No sockets, Python, or game-state writes.
//
// Protocol v2 (2026-11-06): `hello` handshake, `plan2` with a style name, joint-group
// mask and parametric goals; every v1 `plan` line is still accepted unchanged.
#include <motionbricks/inference.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr unsigned joints = 34;
constexpr unsigned stride = joints * 4;
constexpr unsigned protocol_version = 2;
constexpr const char * protocol_family = "skeleto-nn/motion/1";
// Goal plans are a 24-frame in-betweening horizon: four source frames, four target frames.
constexpr unsigned horizon = 24;
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
std::vector<std::string> split(const std::string & value, char separator) {
    std::vector<std::string> parts;
    std::string current;
    for (char ch : value) {
        if (ch == separator) {
            parts.push_back(current);
            current.clear();
        } else current += ch;
    }
    parts.push_back(current);
    return parts;
}

const std::set<std::string> modes{"locomotion", "jump_start", "jump_air", "jump_land", "door", "ladder"};

// Joint groups of the G1 34-joint hierarchy. `head` is accepted for forward compatibility:
// this body has no head joints, so it constrains nothing. `all` reproduces the v1 modes.
const std::map<std::string, std::vector<std::string>> groups = {
    {"legs", {"left_hip_pitch_skel", "left_hip_roll_skel", "left_hip_yaw_skel", "left_knee_skel",
              "left_ankle_pitch_skel", "left_ankle_roll_skel", "left_toe_base",
              "right_hip_pitch_skel", "right_hip_roll_skel", "right_hip_yaw_skel", "right_knee_skel",
              "right_ankle_pitch_skel", "right_ankle_roll_skel", "right_toe_base"}},
    {"spine", {"waist_yaw_skel", "waist_roll_skel", "waist_pitch_skel"}},
    {"arm_l", {"left_shoulder_pitch_skel", "left_shoulder_roll_skel", "left_shoulder_yaw_skel",
               "left_elbow_skel", "left_wrist_roll_skel", "left_wrist_pitch_skel", "left_wrist_yaw_skel",
               "left_hand_roll_skel"}},
    {"arm_r", {"right_shoulder_pitch_skel", "right_shoulder_roll_skel", "right_shoulder_yaw_skel",
               "right_elbow_skel", "right_wrist_roll_skel", "right_wrist_pitch_skel", "right_wrist_yaw_skel",
               "right_hand_roll_skel"}},
    {"head", {}},
};

struct Quaternion { float x{}, y{}, z{}, w{1.0F}; };
using Goals = std::map<std::string, Quaternion>;

Quaternion about_axis(unsigned axis, float angle) {
    Quaternion q{0.0F, 0.0F, 0.0F, std::cos(angle * 0.5F)};
    (axis == 0 ? q.x : axis == 1 ? q.y : q.z) = std::sin(angle * 0.5F);
    return q;
}

// The six legacy modes as goal tables. Values are identical to the v1 keyframes, so a v1
// line produces the same constraints as before: every joint constrained (mask `all`),
// non-goal joints neutral, root height from the mode. `has_root` and `root_y` only apply
// with the full-body mask; partial masks keep the locomotion root.
struct Table { Goals goals; bool has_root{}; float root_y{}; };

Table mode_table(const std::string & mode, float phase) {
    Table table;
    if (mode == "locomotion") return table;
    table.has_root = true;
    table.root_y = mode == "jump_start" ? 0.70F : (mode == "jump_land" ? 0.72F + phase * 0.067193F : 0.787193F);
    if (mode == "door") {
        table.goals["right_shoulder_pitch_skel"] = about_axis(0, -0.9F);
        table.goals["right_elbow_skel"] = about_axis(0, 0.45F);
    } else if (mode == "ladder") {
        for (const std::string side : {"left", "right"}) {
            const float cycle = std::sin(6.2831853F * phase + (side == "left" ? 0.0F : 3.1415927F));
            table.goals[side + "_shoulder_pitch_skel"] = about_axis(0, -1.7F - cycle * 0.3F);
            table.goals[side + "_elbow_skel"] = about_axis(0, 0.7F);
            table.goals[side + "_hip_pitch_skel"] = about_axis(0, -0.45F - cycle * 0.3F);
            table.goals[side + "_knee_skel"] = about_axis(0, 0.85F + cycle * 0.45F);
        }
    } else {
        const float bend = mode == "jump_air" ? 0.85F : (mode == "jump_land" ? 0.55F * (1.0F - phase) : 0.6F);
        for (const std::string side : {"left", "right"}) {
            table.goals[side + "_hip_pitch_skel"] = about_axis(0, -bend * 0.5F);
            table.goals[side + "_knee_skel"] = about_axis(0, bend);
            table.goals[side + "_ankle_pitch_skel"] = about_axis(0, -bend * 0.5F);
            table.goals[side + "_shoulder_pitch_skel"] = about_axis(0, -0.35F);
        }
    }
    return table;
}

struct Request {
    uint64_t id{};
    unsigned advance{};
    float speed{}, mx{}, mz{}, fx{}, fz{}, phase{};
    std::string mode;
    std::string style;        // empty: idle below 0.05 m/s, walk otherwise (v1 behaviour)
    bool full_body{};         // mask `all`: the v1 constrained path
    std::set<std::string> mask;  // constrained joint names (partial mask)
    Goals goals;              // explicit goals, override the mode table
};

struct Runtime {
    mb_runtime_options * options{};
    mb_model * model{};
    mb_agent * agent{};
    mb_command * command{};
    mb_motion * motion{};
    std::map<std::string, mb_style *> styles;
    std::map<std::string, unsigned> indices;
    // Agent context source: the locomotion plan (or the full-body goal motion for `all`).
    std::vector<float> last_roots, last_rotations;
    uint64_t last_frames{};
    // What the game received last; constrained joints continue from here across plans.
    std::vector<float> out_roots, out_rotations;
    uint64_t out_frames{};
    std::set<std::string> out_mask;
    std::string current_mode = "locomotion";

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
        std::vector<std::filesystem::path> files;
        for (const auto & entry : std::filesystem::directory_iterator(directory + "/styles"))
            if (entry.is_regular_file() && entry.path().extension() == ".mbstyle") files.push_back(entry.path());
        std::sort(files.begin(), files.end());
        for (const auto & file : files) {
            mb_style * value{};
            check(mb_style_load(model, file.string().c_str(), &value, error_buffer, sizeof error_buffer));
            styles[file.stem().string()] = value;
        }
        if (!styles.count("idle") || !styles.count("walk")) throw std::runtime_error("bundle lacks idle/walk styles");
        check(mb_agent_create(model, &agent, error_buffer, sizeof error_buffer));
        check(mb_agent_reset(agent, styles.at("idle"), error_buffer, sizeof error_buffer));
        check(mb_command_create(&command, error_buffer, sizeof error_buffer));
        std::cout << "{\"protocol\":\"" << protocol_family << "\",\"protocol_version\":" << protocol_version
                  << ",\"styles\":";
        style_names(std::cout);
        std::cout << ",\"fps\":30,\"threads\":" << threads << ",\"joints\":[";
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
        for (const auto & group : groups)
            for (const auto & name : group.second)
                if (!indices.count(name)) throw std::runtime_error("joint group names a missing joint: " + name);
    }

    void style_names(std::ostream & out) const {
        out << '[';
        bool first = true;
        for (const auto & pair : styles) {
            if (!first) out << ',';
            first = false;
            out << quoted(pair.first);
        }
        out << ']';
    }

    // The configured metres-per-second pace of each exemplar (0 for standing styles).
    void style_speeds(std::ostream & out) const {
        out << '{';
        bool first = true;
        for (const auto & pair : styles) {
            float speed{};
            check(mb_style_get_speed(pair.second, &speed, error_buffer, sizeof error_buffer));
            if (!first) out << ',';
            first = false;
            out << quoted(pair.first) << ':' << speed;
        }
        out << '}';
    }

    void hello(std::ostream & out) const {
        // No "protocol" key here: clients treat that key as the metadata line.
        out << "{\"hello\":true,\"family\":\"" << protocol_family << "\",\"protocol_version\":" << protocol_version
            << ",\"v1\":true,\"styles\":";
        style_names(out);
        out << ",\"style_speeds\":";
        style_speeds(out);
        out << ",\"groups\":{";
        bool first = true;
        for (const auto & group : groups) {
            if (!first) out << ',';
            first = false;
            out << quoted(group.first) << ":[";
            for (std::size_t i = 0; i < group.second.size(); ++i) {
                if (i) out << ',';
                out << quoted(group.second[i]);
            }
            out << ']';
        }
        out << "},\"modes\":[";
        first = true;
        for (const auto & mode : modes) {
            if (!first) out << ',';
            first = false;
            out << quoted(mode);
        }
        out << "]}" << std::endl;
    }

    void context(unsigned advance) {
        if (!last_frames) return;
        const auto first = std::min<uint64_t>(advance, last_frames - 4);
        check(mb_agent_set_context(agent, last_roots.data() + first * 3,
            last_rotations.data() + first * stride, 4, joints, error_buffer, sizeof error_buffer));
    }

    unsigned index(const std::string & name) const {
        const auto found = indices.find(name);
        if (found == indices.end()) throw std::runtime_error("goal joint missing: " + name);
        return found->second;
    }

    static void write(std::vector<float> & rotations, unsigned frame, unsigned joint, const Quaternion & q) {
        float * out = rotations.data() + frame * stride + joint * 4;
        out[0] = q.x; out[1] = q.y; out[2] = q.z; out[3] = q.w;
    }

    // Runs the stateless in-betweener between four source and four target frames and
    // returns the generated motion (24 frames). Goal keyframes are absolute neutral-relative
    // constraints, never offsets compounded onto a prior action.
    mb_motion * inbetween(const std::vector<float> & source_roots, const std::vector<float> & source_rotations,
                          const std::vector<float> & target_roots, const std::vector<float> & target_rotations,
                          uint64_t seed) {
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
            // The upstream masks are per boundary slot (eight per field), not per joint:
            // joint selection happens in the boundary composition above and in compose().
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
        return generated;
    }

    void read_motion(mb_motion * value, std::vector<float> & roots, std::vector<float> & rotations, uint64_t & frames) {
        uint64_t count{};
        const float * data{};
        check(mb_motion_get_frame_count(value, &frames, error_buffer, sizeof error_buffer));
        check(mb_motion_get_root_translations(value, &data, &count, error_buffer, sizeof error_buffer));
        roots.assign(data, data + count);
        check(mb_motion_get_local_rotations_xyzw(value, &data, &count, error_buffer, sizeof error_buffer));
        rotations.assign(data, data + count);
    }

    // v1 path and mask `all`: whole body constrained to neutral + goals at a fixed root height.
    void full_body_goal(const Goals & goals, bool has_root, float root_y, uint64_t seed, float fx, float fz) {
        std::vector<float> plan_roots, plan_rotations;
        uint64_t frames{};
        read_motion(motion, plan_roots, plan_rotations, frames);
        std::vector<float> source_roots(plan_roots.begin(), plan_roots.begin() + 12);
        std::vector<float> source_rotations(plan_rotations.begin(), plan_rotations.begin() + 4 * stride);
        auto target_roots = source_roots;
        std::vector<float> target_rotations(4 * stride, 0.0F);
        const float yaw = std::atan2(fx, fz);
        for (unsigned f = 0; f < 4; ++f) {
            for (unsigned j = 0; j < joints; ++j) target_rotations[f * stride + j * 4 + 3] = 1.0F;
            target_rotations[f * stride + 1] = std::sin(yaw * 0.5F);
            target_rotations[f * stride + 3] = std::cos(yaw * 0.5F);
            if (has_root) target_roots[f * 3 + 1] = root_y;
            for (const auto & goal : goals) write(target_rotations, f, index(goal.first), goal.second);
        }
        mb_motion * generated = inbetween(source_roots, source_rotations, target_roots, target_rotations, seed);
        mb_motion_free(motion);
        motion = generated;
        read_motion(motion, last_roots, last_rotations, last_frames);
        out_roots = last_roots;
        out_rotations = last_rotations;
        out_frames = last_frames;
        out_mask.clear();
    }

    // Partial mask: the locomotion plan keeps every unmasked joint and the root; masked
    // joints are in-betweened from their last delivered pose to the goal (neutral when no
    // goal names them) at the horizon end, then composed over the plan.
    // `release`: the goals are gone; the previously constrained joints blend back into the
    // locomotion plan's own horizon pose instead of snapping to the arm swing.
    void masked_goal(const std::set<std::string> & mask, const Goals & goals, unsigned advance, uint64_t seed,
                     bool release = false) {
        read_motion(motion, last_roots, last_rotations, last_frames);
        if (last_frames < horizon) throw std::runtime_error("locomotion plan shorter than the goal horizon");
        std::vector<float> source_roots(last_roots.begin(), last_roots.begin() + 12);
        std::vector<float> source_rotations(last_rotations.begin(), last_rotations.begin() + 4 * stride);
        const unsigned tail = horizon - 4;
        std::vector<float> target_roots(last_roots.begin() + tail * 3, last_roots.begin() + horizon * 3);
        std::vector<float> target_rotations(last_rotations.begin() + tail * stride,
                                            last_rotations.begin() + horizon * stride);
        // Continuity: a joint constrained in the previous delivery starts from where the
        // game left it (the consumed window), not from the locomotion arm swing.
        const bool continuous = out_frames >= 4 && !out_mask.empty();
        const unsigned from = continuous ? static_cast<unsigned>(std::min<uint64_t>(advance, out_frames - 4)) : 0;
        for (unsigned f = 0; f < 4; ++f) {
            for (const auto & name : mask) {
                const unsigned j = index(name);
                if (continuous && out_mask.count(name))
                    std::copy_n(out_rotations.data() + (from + f) * stride + j * 4, 4,
                                source_rotations.data() + f * stride + j * 4);
                if (release) continue;
                const auto goal = goals.find(name);
                write(target_rotations, f, j, goal == goals.end() ? Quaternion{} : goal->second);
            }
        }
        mb_motion * generated = inbetween(source_roots, source_rotations, target_roots, target_rotations, seed);
        std::vector<float> generated_roots, generated_rotations;
        uint64_t generated_frames{};
        try {
            read_motion(generated, generated_roots, generated_rotations, generated_frames);
        } catch (...) {
            mb_motion_free(generated);
            throw;
        }
        mb_motion_free(generated);
        if (generated_frames < horizon) throw std::runtime_error("goal motion shorter than the horizon");
        out_frames = horizon;
        out_roots.assign(last_roots.begin(), last_roots.begin() + horizon * 3);
        out_rotations.assign(last_rotations.begin(), last_rotations.begin() + horizon * stride);
        for (unsigned f = 0; f < horizon; ++f)
            for (const auto & name : mask) {
                const unsigned j = index(name);
                std::copy_n(generated_rotations.data() + f * stride + j * 4, 4, out_rotations.data() + f * stride + j * 4);
            }
        out_mask = release ? std::set<std::string>{} : mask;
    }

    void plan(const Request & request) {
        // plan2 may send SPEED -1: the style's configured pace (0.3 m/s for slow_walk).
        const bool style_pace = request.speed == -1.0F && !request.style.empty();
        if (request.advance > 64 || (request.speed < 0 && !style_pace) || request.speed > 10 ||
            request.phase < 0 || request.phase > 1 ||
            !std::isfinite(request.speed + request.mx + request.mz + request.fx + request.fz + request.phase) ||
            std::hypot(request.fx, request.fz) < 0.001F || std::hypot(request.mx, request.mz) < 0.001F)
            throw std::runtime_error("invalid movement request");
        if (!modes.count(request.mode)) throw std::runtime_error("unknown motion goal");
        const bool automatic = request.style.empty();
        if (!automatic && !styles.count(request.style)) throw std::runtime_error("unknown style: " + request.style);
        for (const auto & goal : request.goals) index(goal.first);
        Table table = mode_table(request.mode, request.phase);
        for (const auto & goal : request.goals) table.goals[goal.first] = goal.second;
        const bool full_body = request.full_body || (request.mode != "locomotion" && request.mask.empty() && request.goals.empty());
        std::set<std::string> mask = request.mask;
        if (!full_body) for (const auto & goal : table.goals) mask.insert(goal.first);
        if (request.mode != current_mode) {
            check(mb_agent_reset(agent, styles.at("idle"), error_buffer, sizeof error_buffer));
            last_frames = 0;
            last_roots.clear();
            last_rotations.clear();
            out_frames = 0;
            out_mask.clear();
            current_mode = request.mode;
        }
        context(request.advance);
        mb_motion_free(motion);
        motion = nullptr;
        const auto started = std::chrono::steady_clock::now();
        // Standing still always uses the idle style: the walking styles shuffle at zero speed.
        const std::string style_name = style_pace ? request.style
            : (request.speed < 0.05F ? "idle" : (automatic ? "walk" : request.style));
        check(mb_command_set_style(command, styles.at(style_name), error_buffer, sizeof error_buffer));
        check(mb_command_set_target_speed(command, request.speed, error_buffer, sizeof error_buffer));
        check(mb_command_set_movement_direction(command, request.mx, 0, request.mz, error_buffer, sizeof error_buffer));
        check(mb_command_set_facing_direction(command, request.fx, 0, request.fz, error_buffer, sizeof error_buffer));
        check(mb_command_set_seed(command, request.id, error_buffer, sizeof error_buffer));
        check(mb_agent_plan(agent, command, &motion, error_buffer, sizeof error_buffer));
        if (full_body) full_body_goal(table.goals, table.has_root, table.root_y, request.id, request.fx, request.fz);
        else if (!mask.empty()) masked_goal(mask, table.goals, request.advance, request.id);
        else if (!out_mask.empty()) masked_goal(out_mask, {}, request.advance, request.id, true);
        else {
            read_motion(motion, last_roots, last_rotations, last_frames);
            out_roots = last_roots;
            out_rotations = last_rotations;
            out_frames = last_frames;
            out_mask.clear();
        }
        for (float value : out_roots) if (!std::isfinite(value)) throw std::runtime_error("non-finite root");
        for (float value : out_rotations) if (!std::isfinite(value)) throw std::runtime_error("non-finite pose");
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        std::cout << "{\"id\":" << request.id << ",\"mode\":" << quoted(request.mode)
                  << ",\"style\":" << quoted(style_name) << ",\"masked\":" << mask.size()
                  << ",\"frames\":" << out_frames << ",\"inference_ms\":" << ms << ",\"roots\":";
        array(std::cout, out_roots.data(), out_roots.size());
        std::cout << ",\"rotations\":";
        array(std::cout, out_rotations.data(), out_rotations.size());
        std::cout << '}' << std::endl;
    }
};

// MASK: `none`, `all`, or a comma list of group names. GOALS: `-` or `joint=qx,qy,qz,qw;...`.
void parse_v2(Request & request, const std::string & style, const std::string & mask, const std::string & goals) {
    if (style != "-") request.style = style;
    if (mask == "all") request.full_body = true;
    else if (mask != "none") {
        for (const auto & group : split(mask, ',')) {
            const auto found = groups.find(group);
            if (found == groups.end()) throw std::runtime_error("unknown joint group: " + group);
            request.mask.insert(found->second.begin(), found->second.end());
        }
    }
    if (goals == "-") return;
    for (const auto & item : split(goals, ';')) {
        const auto equals = item.find('=');
        if (equals == std::string::npos || equals == 0) throw std::runtime_error("malformed goal: " + item);
        const auto values = split(item.substr(equals + 1), ',');
        if (values.size() != 4) throw std::runtime_error("goal needs four quaternion components: " + item);
        std::array<float, 4> q{};
        for (unsigned i = 0; i < 4; ++i) {
            std::size_t used{};
            try { q[i] = std::stof(values[i], &used); } catch (const std::exception &) { used = 0; }
            if (used != values[i].size() || !std::isfinite(q[i])) throw std::runtime_error("malformed goal: " + item);
        }
        const float norm = std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
        if (std::abs(norm - 1.0F) > 0.01F) throw std::runtime_error("goal quaternion is not unit: " + item);
        request.goals[item.substr(0, equals)] = Quaternion{q[0]/norm, q[1]/norm, q[2]/norm, q[3]/norm};
    }
}
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
                std::string operation, extra;
                input >> operation;
                if (operation == "hello" || operation == "version") {
                    if (input >> extra) throw std::runtime_error("malformed request");
                    runtime.hello(std::cout);
                    continue;
                }
                Request request;
                if (!(input >> request.id >> request.advance >> request.speed >> request.mx >> request.mz
                      >> request.fx >> request.fz >> request.mode >> request.phase))
                    throw std::runtime_error("malformed request");
                if (operation == "plan") {
                    request.full_body = request.mode != "locomotion";
                } else if (operation == "plan2") {
                    std::string style, mask, goals;
                    if (!(input >> style >> mask >> goals)) throw std::runtime_error("malformed plan2 request");
                    parse_v2(request, style, mask, goals);
                } else throw std::runtime_error("malformed request");
                if (input >> extra) throw std::runtime_error("malformed request");
                runtime.plan(request);
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
