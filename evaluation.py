import pandas as pd
import nltk.translate.bleu_score as bleu

class EvaluationModel:
    base_path = "./validation_result/postprocessed"
    models = ["LEmodel_2", "LEmodel_3", "model1", "model2","model3", "llama"]

    def load_result(self,model_name):
        df = pd.read_csv(f"{self.base_path}/{model_name}_validation_result.csv")
        return df[['example','model']]

    def bleu(self, candidate, reference):
        return bleu.sentence_bleu(list(map(lambda ref: ref.split(), [reference])), candidate.split())

    def bleu_evaluation_models(self):
        for model in self.models:
            validation_result = self.load_result(model)
            result = validation_result.apply(lambda x: self.bleu(x['example'], x['model']), axis=1)
            print(result)


EvaluationModel().bleu_evaluation_models()