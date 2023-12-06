import pandas as pd
import re
import ast

files = ["LEmodel_2_validation_result.csv",
         "LEmodel_3_validation_result.csv"]

files2 = ["model1_validation_result.csv",
          "model2_validation_result.csv",
          "model3_validation_result.csv"]


# for file in ["llama_validation_result.csv"]:
#     df = pd.read_csv(file, index_col=0)
#     df['llama'] = df.apply(lambda x: x["llama"].replace(f"{x['prompt']}", "").strip(), axis=1)
#     df.to_csv(f"./postprocessed/{file}")

def split_example(text):
    for pt in [r"<s>\[INST\]", r"</s>"]:
        text = re.sub(pt, "", text)
    print(text)
    result = re.split(r"\[/INST]", text)
    return result[1].strip() if len(result) > 1 else result[0].strip()


# for file in files: v
#     df = pd.read_csv(file, index_col=0)
#     df['model'] = df['model'].apply(lambda result: split_example(result))
#     df.to_csv(f"./postprocessed/{file}")

for file in files2:
    df = pd.read_csv(file, index_col=0)
    remove_shell = lambda x: re.sub("}]", "", re.sub(r"\[{'generated_text': [\"\']", "", x)).strip()
    df['model'] = df['model'].apply(lambda result:remove_shell(result))
    df['model'] = df['model'].apply(lambda result: split_example(result))

    df.to_csv(f"./postprocessed/{file}")
