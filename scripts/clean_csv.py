import pandas as pd

df = pd.read_csv("../data/annotated_frames.csv")
df.drop(columns=["ID", "ENGLISH", "FRAME1", "FRAME2", "FRAME3"], inplace=True)
print(df.info())

df.to_csv("clean_annotated_frames.csv", index=False)