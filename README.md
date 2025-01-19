# Arabic Handwritten Character Recognition

A deep learning application that recognizes handwritten Arabic characters using an Artificial Neural Network (ANN). The project includes a FastAPI backend and Streamlit frontend for easy interaction.

## Features

- Recognition of 28 Arabic characters
- Real-time prediction through web interface
- Simple and intuitive UI
- REST API backend
- High accuracy prediction

## Project Structure

project/
├── backend/
│ └── main.py
├── frontend/
│ └── streamlit_app.py
├── model/
│ ├── arabic_handwritten.h5
│ └── label_encoder.pkl
├── main.py
├── requirements.txt
├── README.md
└── .gitignore


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/arabic-handwritten-recognition.git
   ```


2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```


3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```


## Usage

1. Train the model (if not already trained):
   ```bash
   python main.py
   ```


2. Start the FastAPI backend:
   ```bash
   uvicorn backend.main:app --reload
   ```


3. In a new terminal, start the Streamlit frontend:
   ```bash
   streamlit run frontend/streamlit_app.py
   ```


4. Open your web browser and navigate to `http://localhost:8501` to use the application.

## Dataset

This project uses the Arabic Handwritten Characters Dataset (AHCD) from Kaggle, containing 16,800 characters written by different users.

## Model Architecture

The ANN model consists of:
- Input layer: 1024 neurons (32x32 flattened images)
- Hidden layers:
  - Dense layer (512 neurons, ReLU activation)
  - Dense layer (256 neurons, ReLU activation)
  - Dense layer (128 neurons, ReLU activation)
- Output layer: 28 neurons (softmax activation)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [AHCD Dataset](https://www.kaggle.com/mloey1/ahcd1) by MLOEY
- TensorFlow and Keras teams
- FastAPI and Streamlit communities