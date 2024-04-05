from utils import getExistingProfileName
import pandas as pd
import numpy as np

def test_answer():
    data = {"UserId" : [1, 1, 1, 2, 3, 4], "ProfileName" : ["Bubbles", "Bubbles", np.nan, "Fox", "Bird", ""]}
    df = pd.DataFrame(data)
    assert getExistingProfileName(df, 1) == "Bubbles"