import os
import logging
from utils.utils import prediction

logger = logging.getLogger(__name__)


class Runner:

    def __init__(self, model_name, trainer, tokenizer, training_args, test=None, eval=None):
        self.model_name = model_name
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.training_args = training_args
        self.test = test
        self.dev = eval

    def __call__(self):
        if self.training_args.do_train:
            self.train(self.model_name)

        if self.training_args.do_eval and self.dev is not None:
            self.eval(self.dev)

        if self.training_args.do_predict and self.test is not None:
            self.predict(self.test)

    def train(self, model_name):
        self.trainer.train(model_path=model_name if os.path.isdir(model_name) else None)
        self.trainer.save_model()
        if self.trainer.is_world_process_zero():
            self.tokenizer.save_pretrained(self.training_args.output_dir)

    def eval(self,eval):
        logger.info("*** Evaluate ***")
        result = self.trainer.evaluate(eval_dataset=eval)
        output_eval_file = os.path.join(self.training_args.output_dir, "eval_results.txt")
        if self.trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in result.items():
                    logger.info("%s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

        logger.info("Validation set result : {}".format(result))

    def predict(self, test):
        logger.info("*** Test ***")
        predictions = self.trainer.predict(test_dataset=test)
        output_test_file = os.path.join(self.training_args.output_dir, "test_results.txt")
        if self.trainer.is_world_process_zero():
            with open(output_test_file, "w") as writer:
                logger.info("***** Test results *****")
                logger.info("{}".format(predictions))
                writer.write("prediction : \n{}\n\n".format(prediction(predictions.predictions).tolist()))
                if predictions.label_ids is not None:
                    writer.write("ground truth : \n{}\n\n".format(predictions.label_ids.tolist()))
                    writer.write("metrics : \n{}\n\n".format(predictions.metrics))
