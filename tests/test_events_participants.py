"""viewer/events.py participant parsing on SYNTHETIC wikitext: a team Liquipedia
moved to {{FormerParticipants}} (withdrew / replaced) is not a participant."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402,F401

import events as E  # noqa: E402

PAGE = """
{{TeamParticipants
|{{Opponent|Alpha
|players={{Persons
|{{Person|a1}}
|{{Person|role=coach|ac}}
}}
}}
|{{Opponent|Bravo
|players={{Persons
|{{Person|b1}}
}}
|notes={{Notes|{{Notes|October 9th - '''[[Charlie]]''' withdraw; replaced by '''[[Bravo]]'''.}}}}
}}
}}
{{FormerParticipants
|{{TeamParticipants
|{{Opponent|Charlie
|players={{Persons
|{{Person|c1}}
}}
}}
}}
}}
"""


class FormerParticipantsTest(unittest.TestCase):
    def test_former_participant_left_out(self):
        got = E.parse_participants(PAGE, {})
        self.assertEqual(sorted(got), ["Alpha", "Bravo"])
        self.assertEqual(got["Alpha"]["coach"], ["ac"])


if __name__ == "__main__":
    unittest.main()
