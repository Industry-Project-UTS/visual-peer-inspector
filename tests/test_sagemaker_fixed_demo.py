import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('fixed_demo', Path(__file__).resolve().parents[1] / 'apps/web/sagemaker/fixed_demo.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)

class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'settings.json'
        self.settings = {'public_url':'https://example.ngrok-free.dev', 'username':'test-user', 'password':'test-only-password-1234'}
    def save(self, **changes):
        self.path.write_text(json.dumps({**self.settings, **changes}))
        self.path.chmod(0o600)
    def test_private_settings(self):
        self.save()
        self.assertEqual(helper.read_settings(self.path)['public_url'], self.settings['public_url'])
    def test_rejects_weak_or_missing_auth(self):
        for updates in ({'password':''}, {'password':'abc'}, {'username':''}):
            self.save(**updates)
            with self.assertRaises(ValueError):helper.read_settings(self.path)
    def test_rejects_unexpected_domain_or_url_credentials(self):
        for url in ['http://example.ngrok-free.dev','https://example.com','https://u:p@example.ngrok-free.dev','https://example.ngrok-free.dev.evil.test','https://example.ngrok-free.dev/path','https://example.ngrok-free.dev?token=x']:
            self.save(public_url=url)
            with self.assertRaises(ValueError):helper.read_settings(self.path)
    def test_rejects_publicly_readable_secret(self):
        self.save();self.path.chmod(0o644)
        with self.assertRaises(ValueError):helper.read_settings(self.path)
    def test_rejects_symlink(self):
        self.save();link=self.path.with_name('linked.json');link.symlink_to(self.path)
        with self.assertRaises(ValueError):helper.read_settings(link)

if __name__=='__main__':unittest.main()
