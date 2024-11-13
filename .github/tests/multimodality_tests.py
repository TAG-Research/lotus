import os

import pandas as pd
import pytest

import lotus
from lotus.models import LM
from lotus.dtype_extensions import ImageArray

################################################################################
# Setup
################################################################################
# Set logger level to DEBUG
lotus.logger.setLevel("DEBUG")

# Environment flags to enable/disable tests
ENABLE_OPENAI_TESTS = os.getenv("ENABLE_OPENAI_TESTS", "false").lower() == "true"

MODEL_NAME_TO_ENABLED = {
    "gpt-4o-mini": ENABLE_OPENAI_TESTS,
    "gpt-4o": ENABLE_OPENAI_TESTS,
}
ENABLED_MODEL_NAMES = set([model_name for model_name, is_enabled in MODEL_NAME_TO_ENABLED.items() if is_enabled])


def get_enabled(*candidate_models: str) -> list[str]:
    return [model for model in candidate_models if model in ENABLED_MODEL_NAMES]


@pytest.fixture(scope="session")
def setup_models():
    models = {}

    for model_path in ENABLED_MODEL_NAMES:
        models[model_path] = LM(model=model_path)

    return models


@pytest.fixture(autouse=True)
def print_usage_after_each_test(setup_models):
    yield  # this runs the test
    models = setup_models
    for model_name, model in models.items():
        print(f"\nUsage stats for {model_name} after test:")
        model.print_total_usage()
        model.reset_stats()


################################################################################
# Standard tests
################################################################################
@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini"))
def test_filter_operation(setup_models, model):
    lm = setup_models[model]
    lotus.settings.configure(lm=lm)

    # Test filter operation on an easy dataframe
    image_url = [
        "https://img.etsystatic.com/il/4bee20/1469037676/il_340x270.1469037676_iiti.jpg?version=0",
        "https://thumbs.dreamstime.com/b/comida-r%C3%A1pida-nachos-con-el-sause-del-tomate-ejemplo-exhausto-de-la-acuarela-mano-aislado-en-blanco-150936354.jpg",
        "https://i1.wp.com/www.alloverthemap.net/wp-content/uploads/2014/02/2012-09-25-12.46.15.jpg?resize=400%2C284&amp;ssl=1",
        "https://i.pinimg.com/236x/a4/3a/65/a43a65683a0314f29b66402cebdcf46d.jpg",
        "https://pravme.ru/wp-content/uploads/2018/01/sobor-Bogord-1.jpg"
    ]
    df = pd.DataFrame({"image": ImageArray(image_url)})
    user_instruction = "{image} represents food"
    filtered_df = df.sem_filter(user_instruction)

    assert image_url[1] in filtered_df["image"].values