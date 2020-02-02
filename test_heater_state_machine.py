import unittest

from heater_state_machine import HeaterState


class HeaterStateTest(unittest.TestCase):
    def test_simple(self):
        heater = HeaterState(2)

        self.assertEqual(heater.minState + 1, heater.maxState)

        heater.on_event(0)
        self.assertEqual(heater.state, heater.minState)
        self.assertAlmostEqual(heater.on_event(1), 1)
        self.assertEqual(heater.state, heater.maxState)
        self.assertAlmostEqual(heater.on_event(1), 1)
        self.assertEqual(heater.state, heater.maxState)
        heater.on_event(0)
        self.assertEqual(heater.state, heater.minState)

    def test_extended(self):
        heater = HeaterState(4)

        self.assertEqual(heater.minState + 3, heater.maxState)

        heater.on_event(0)
        self.assertEqual(heater.state, heater.minState)
        self.assertAlmostEqual(heater.on_event(1), 1./3)
        self.assertEqual(heater.state, heater.minState + 1)
        self.assertAlmostEqual(heater.on_event(0), 0)
        self.assertEqual(heater.state, heater.minState)
        for i in range(3):
            self.assertAlmostEqual(heater.on_event(1), (i + 1) / 3.)
            self.assertEqual(heater.state, heater.minState + i + 1)

        self.assertEqual(heater.state, heater.maxState)

        for i in range(3):
            self.assertAlmostEqual(heater.on_event(0), max(heater.maxState - i - 1, 0) / 3.)
            self.assertEqual(heater.state, heater.maxState - i - 1)

        self.assertEqual(heater.state, heater.minState)


if __name__ == '__main__':
    unittest.main()
