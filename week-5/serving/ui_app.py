import streamlit as st
from serving.predictor import Predictor
from PIL import Image


@st.cache(hash_funcs={Predictor: lambda _: None})
def load_model() -> Predictor:
    return Predictor.default_from_model_registry()


model = load_model()


def main():
    st.header("MNIST serving")
    image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    btn_predict = st.button('Predict')

    if image_file is not None:
        image = Image.open(image_file).convert('RGB')
        st.image(image, caption='Your image')

        if btn_predict:
            output = model.predict(image)
            st.write(f'Your output is: {output}')


if __name__ == '__main__':
    main()
