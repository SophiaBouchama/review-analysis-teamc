# Utils functions
def getExistingProfileName(df, userId):
    newProfileName = "Anonymous"
    profileNames = df[df.UserId == userId].ProfileName.dropna()

    if len(profileNames):
        newProfileName = profileNames[0]

    return newProfileName