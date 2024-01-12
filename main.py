import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Hash import SHA256


class CheetahKeyGenerator:

    def __init__(self, image):
        self.image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def isolate_spots(self, threshold_value, max_value):
        _, binary_thresh = cv2.threshold(self.gray_image, threshold_value,
                                         max_value, cv2.THRESH_BINARY_INV)
        return binary_thresh

    def find_and_filter_contours(self, binary_thresh, min_contour_area,
                                 max_contour_area):
        contours, _ = cv2.findContours(binary_thresh, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
        return [
            contour for contour in contours
            if cv2.contourArea(contour) > min_contour_area
            and cv2.contourArea(contour) <= max_contour_area
        ]

    def apply_edge_detection(self, low_threshold, high_threshold):
        # Canny edge detection
        return cv2.Canny(self.gray_image, low_threshold, high_threshold)

    def morphological_operations(self, operation, kernel_size):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if operation == 'Dilation':
            return cv2.dilate(self.gray_image, kernel, iterations=1)
        elif operation == 'Erosion':
            return cv2.erode(self.gray_image, kernel, iterations=1)
        elif operation == 'Opening':
            return cv2.morphologyEx(self.gray_image, cv2.MORPH_OPEN, kernel)
        elif operation == 'Closing':
            return cv2.morphologyEx(self.gray_image, cv2.MORPH_CLOSE, kernel)
        else:
            return self.gray_image

    def generate_key_visualization(self, contours):
        areas = [cv2.contourArea(c) for c in contours]
        # Filter out areas that are too large (outliers)
        max_area_threshold = 5000  # Set a threshold value
        filtered_areas = [area for area in areas if area < max_area_threshold]

        fig, ax = plt.subplots()
        ax.hist(filtered_areas, bins=20, color='blue', alpha=0.7)
        ax.set_title('Distribution of Contour Areas (Filtered)')
        ax.set_xlabel('Area')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

        # Now proceed with key generation
        areas_bytes = b''.join(
            int(area).to_bytes((int(area).bit_length() + 7) // 8, 'big')
            for area in filtered_areas if area > 0)
        hash_obj = SHA256.new(areas_bytes)
        return hash_obj.digest()

    def create_image_from_contours(self, contours):
        mask = np.zeros(self.gray_image.shape, dtype=np.uint8)
        cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
        return mask


def encrypt_message(key, message):
    cipher = AES.new(key, AES.MODE_CBC)
    ct_bytes = cipher.encrypt(pad(message.encode(), AES.block_size))
    return cipher.iv + ct_bytes


def decrypt_message(key, ciphertext):
    iv = ciphertext[:AES.block_size]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    pt = unpad(cipher.decrypt(ciphertext[AES.block_size:]), AES.block_size)
    return pt.decode()


@st.cache_data
def process_image(uploaded_file, threshold_value, max_value, min_contour_area,
                  max_contour_area, edge_detection, low_threshold,
                  high_threshold, morph_op, kernel_size, use_processed_image):
    generator = CheetahKeyGenerator(Image.open(uploaded_file))

    processed_image = generator.gray_image
    if edge_detection:
        edges = generator.apply_edge_detection(low_threshold, high_threshold)
        st.image(edges, caption='Advanced Edge Detection Applied')
        if use_processed_image:
            processed_image = edges

    morphed_image = generator.morphological_operations(morph_op, kernel_size)
    st.image(morphed_image, caption='Morphological Operation Applied')
    if morph_op != 'None' and use_processed_image:
        processed_image = morphed_image

    if use_processed_image:
        generator.gray_image = processed_image

    # Update the isolate_spots and find_and_filter_contours methods to use processed_image
    binary_thresh = generator.isolate_spots(
        threshold_value,
        max_value,
    )
    contours = generator.find_and_filter_contours(binary_thresh,
                                                  min_contour_area,
                                                  max_contour_area)

    st.subheader("Visualizing Key Generation")
    key = generator.generate_key_visualization(contours)
    return key, generator.gray_image, binary_thresh, generator.create_image_from_contours(
        contours)


def main():
    st.sidebar.title('Settings')
    threshold_value = st.sidebar.slider('Binary Threshold', 0, 255, 175)
    max_value = st.sidebar.slider('Max Binary Value', 0, 255, 255)
    min_contour_area = st.sidebar.slider('Minimum Contour Area (px)', 0, 1000,
                                         30)
    max_contour_area = st.sidebar.slider('Maximum Contour Area (px)',
                                         min_contour_area, 10000, 3000)
    edge_detection = st.sidebar.checkbox('Apply Edge Detection')
    low_threshold = st.sidebar.slider('Low Threshold for Edge Detection', 0,
                                      100, 50)
    high_threshold = st.sidebar.slider('High Threshold for Edge Detection',
                                       101, 200, 150)
    morph_op = st.sidebar.selectbox(
        'Morphological Operation',
        ['None', 'Dilation', 'Erosion', 'Opening', 'Closing'])
    kernel_size = st.sidebar.slider('Kernel Size for Morphological Operations',
                                    1, 10, 3)

    uploaded_file = st.sidebar.file_uploader("Choose a cheetah image...",
                                             type=['png', 'jpg', 'jpeg'])

    use_processed_image = st.sidebar.checkbox(
        "Use Processed Image for Key Generation")

    st.title('Cheetah Cryptographic Key Generator')
    st.markdown(
        """ ## Overview This application generates a cryptographic key from the unique patterns of a cheetah's coat. The generated key is then used to encrypt and decrypt messages. """
    )

    if uploaded_file is not None:
        key, gray_image, binary_thresh, spots_image = process_image(
            uploaded_file, threshold_value, max_value, min_contour_area,
            max_contour_area, edge_detection, low_threshold, high_threshold,
            morph_op, kernel_size, use_processed_image)
        st.subheader("Image Processing Steps")
        st.image(gray_image, caption='Grayscale Image')
        st.image(binary_thresh, caption='Binary Threshold Applied')
        st.image(spots_image, caption='Filtered Contours')

        st.subheader("Generated Cryptographic Key")
        st.code(key.hex(), language='python')

        user_input = st.text_area("Enter a message to encrypt",
                                  "Hello Streamlit",
                                  height=100)
        if st.button('Encrypt'):
            with st.spinner('Encrypting...'):
                encrypted_message = encrypt_message(key, user_input)
                st.text_area("Encrypted Message",
                             encrypted_message.hex(),
                             height=100,
                             key="encrypted_message")
                decrypted_message = decrypt_message(key, encrypted_message)
                st.text_area("Decrypted Message",
                             decrypted_message,
                             height=100,
                             key="decrypted_message")

                if user_input == decrypted_message:
                    st.success(
                        "The encryption and decryption process was successful."
                    )
                else:
                    st.error(
                        "There was an error in the encryption/decryption process."
                    )


if __name__ == '__main__':
    main()
