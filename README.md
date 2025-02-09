# CV-segmentation

This repository contains code for an application that performs automatic inference using a fine-tuned segmentation model to segment images uploaded by the user.

## Features

- **Image Upload**: Users can upload images for segmentation.
- **Automatic Inference**: The app utilizes a fine-tuned segmentation model to process and segment the uploaded images.
- **User-Friendly Interface**: A simple and intuitive interface for seamless user experience.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/DamiFass/CV-segmentation.git
   cd CV-segmentation
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Application**:

   ```bash
   Streamlit run stApp.py
   ```

2. **Access the App**:

   Open your web browser to interact with the application.

3. **Upload and Segment Images**:

   - Click on the "Upload Image" button to select an image from your device.
   - The app will automatically perform segmentation on the uploaded image and display the result.

4. **Access Streamlit app directly**:

    Alternatively, one can try the Streamlit app already deployed [here](https://find-people-easy.streamlit.app/).

## Project Structure

- `src/`: Contains the source code for the segmentation model and related utilities.
- `stApp.py`: The main script to run the Streamlit application.
- `requirements.txt`: Lists the Python dependencies required for the project.

## Dependencies

The application relies on several Python libraries, including but not limited to:

- `streamlit`: For building the web application interface.
- `torch`: For loading and running the segmentation model.
- `PIL`: For image processing.

For a complete list of dependencies, refer to the `requirements.txt` file.

## Contributing

Contributions are welcome! If you have suggestions or improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. 
