import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge import Rouge
import sys
sys.setrecursionlimit(100000)



class EvaluationModel:
    base_path = "./validation_result/postprocessed"
    models = ["llama",
              "LEmodel_2",
              "LEmodel_3",
              "model1",
              "model3",
              "model2"]

    def load_result(self, model_name):
        df = pd.read_csv(f"{self.base_path}/{model_name}_validation_result.csv")
        return df[['example', 'model']]

    def bleu(self, candidate, reference):
        cand = str(candidate).split()
        ref = list(map(lambda ref: str(ref).split(), [reference]))
        return sentence_bleu(ref, cand, smoothing_function=SmoothingFunction().method2)

    def rouge(self, hypothesis, reference):
        rouge_scorer = Rouge()
        score = rouge_scorer.get_scores(
            hyps=str(hypothesis),
            refs=str(reference),
        )

        return score[0]["rouge-l"]["f"]

    def evaluation_all_models(self):
        score_result = pd.DataFrame(columns=['name', 'bleu','rouge'])
        for model in self.models:
            validation_result = self.load_result(model)
            bleu = validation_result.apply(lambda x: self.bleu(x['example'], x['model']), axis=1).mean()
            bleu = round(bleu, 5)
            rouge = validation_result.apply(lambda x: self.rouge(x['example'], x['model']), axis=1).mean()
            rouge = round(rouge, 5)
            data = pd.DataFrame({"name": model, "bleu": [bleu], 'rouge':[rouge]})
            score_result = pd.concat([score_result, data])
        score_result.to_csv("scores.csv", index=False)

EvaluationModel().evaluation_all_models()
