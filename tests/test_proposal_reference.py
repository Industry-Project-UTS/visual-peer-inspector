import unittest
from io import BytesIO

from PIL import Image

from visual_peer_inspector.evaluation.proposal_reference import _mask_geometry


class ProposalReferenceTests(unittest.TestCase):
    @staticmethod
    def mask_bytes(value: int, box=None) -> bytes:
        image = Image.new("L", (256, 256), color=0)
        if box is not None:
            pixels = image.load()
            for y in range(box[1], box[3]):
                for x in range(box[0], box[2]):
                    pixels[x, y] = value
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def test_mask_threshold_returns_half_open_box(self):
        box, area = _mask_geometry(self.mask_bytes(205, [3, 4, 8, 10]), 128)

        self.assertEqual(box, [3, 4, 8, 10])
        self.assertEqual(area, 30)

    def test_background_values_below_threshold_are_not_objects(self):
        box, area = _mask_geometry(self.mask_bytes(1, [0, 0, 256, 256]), 128)

        self.assertIsNone(box)
        self.assertEqual(area, 0)


if __name__ == "__main__":
    unittest.main()
