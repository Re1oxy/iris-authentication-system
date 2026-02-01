import unittest
from core.auth_system import AuthSystem
from core.data_loader import DataLoader

class TestAuthenticationSystem(unittest.TestCase):
    def test_authenticate_output(self):
        system = AuthSystem()
        loader = DataLoader(
            train_path="dataset/database.csv",
            test_path="dataset/test_dataset.csv",
            label_column=0
        )
        X_test, _ = loader.load_test()
        sample = X_test.iloc[0].to_numpy().reshape(1, -1)
        decision, confidence = system.authenticate(sample)


        self.assertIn(decision, ["granted", "denied_low_confidence", "denied_spoof"])
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

if __name__ == "__main__":
    unittest.main()
