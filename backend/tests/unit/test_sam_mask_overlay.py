import json
import unittest
from pathlib import Path
from app.perception.sam_mask_overlay import parse_image_records


class SamMaskOverlayTests(unittest.TestCase):
    def test_parses_lossless_and_one_bit(self):
        raw = ('<0f>0<|box;x1=1;y1=2;x2=2;y2=3;w=20;h=10;c=0.9|>'
               '<|mask;x=0;y=0;data=2,2,~abc|>,'
               '1<|box;x1=4;y1=4;x2=5;y2=5;w=20;h=10|>'
               '<|mask;x=0;y=0;data=2,2,!abc|>')
        records = parse_image_records(raw)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]['encoding'], 'lossless')
        self.assertEqual(records[1]['encoding'], 'one_bit')
        self.assertEqual(records[0]['confidence'], 0.9)

    def test_rejects_bad_dimensions(self):
        raw = '<0f>0<|box;x1=1;y1=2;x2=3;y2=3;w=20;h=10|><|mask;x=0;y=0;data=2,2,~abc|>'
        with self.assertRaises(ValueError):
            parse_image_records(raw)

    def test_rejects_trailing_garbage(self):
        with self.assertRaises(ValueError):
            parse_image_records('<0f>not a record')

    def test_empty_result(self):
        self.assertEqual(parse_image_records('<0f>'), [])


if __name__ == '__main__':
    unittest.main()
