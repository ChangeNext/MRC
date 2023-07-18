from transformers import AutoModelForQuestionAnswering, AutoTokenizer, AutoModel, AutoConfig

def load_model(model_name):
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForQuestionAnswering.from_pretrained(model_name)

	return tokenizer, model

def test_load_model(model_path):
	model = AutoModelForQuestionAnswering.from_pretrained(model_path)

	return model

