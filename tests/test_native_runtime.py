import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from runtime.package import package


class NativePackagingTests(unittest.TestCase):
    def make_bundle(self, root):
        root.mkdir()
        files = {'skeleto_motion_worker': b'native helper', 'g1-f32/pose.gguf': b'model fixture',
                 'UPSTREAM_LICENSE': b'upstream license', 'NOTICE': b'attribution'}
        manifest = {'protocol': 'skeleto-nn/motion/1', 'files': {}}
        for name, content in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            manifest['files'][name] = {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
        (root / 'runtime.json').write_text(json.dumps(manifest))
        return manifest

    def test_relocatable_macos_bundle_keeps_notices(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_bundle(root / 'source')
            app = root / 'Game.app'
            (app / 'Contents/MacOS').mkdir(parents=True)
            result = package(root / 'source', app)
            self.assertEqual(result, app.resolve() / 'Contents/Resources/neural_runtime')
            for name in ['runtime.json', 'skeleto_motion_worker', 'UPSTREAM_LICENSE', 'NOTICE']:
                self.assertTrue((result / name).is_file())

    def test_corruption_is_rejected_before_destination_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_bundle(root / 'source')
            (root / 'source/g1-f32/pose.gguf').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                package(root / 'source', root / 'output')
            self.assertFalse((root / 'output').exists())

    def test_bundle_cannot_write_outside_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.make_bundle(root / 'source')
            manifest['files']['../escape'] = {'bytes': 0, 'sha256': ''}
            (root / 'source/runtime.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                package(root / 'source', root / 'output')
            self.assertFalse((root / 'escape').exists())


if __name__ == '__main__':
    unittest.main()
