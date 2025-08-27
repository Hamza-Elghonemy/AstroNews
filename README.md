# AstroNews

A modern NLP-powered application for analyzing and exploring space news articles. AstroNews leverages state-of-the-art embedding models to provide deep semantic understanding of textual content.

## 🚀 Features

* **Semantic Search**: Find articles based on meaning and context, not just keywords.
* **Article Clustering**: Automatically group similar news articles together.
* **Content Summarization**:  Get the gist of an article without reading the whole text.

## 🛠️ Technology Stack

* **Language**: Python
* **Core Libraries**:
  * `transformers`: For accessing and using state-of-the-art NLP models.
  * `torch` / `tensorflow`: As the backend for the transformer models.
  * `scikit-learn`: For machine learning tasks like clustering.
  * `pandas`: For efficient data manipulation.
* **Embedding Model**: `BAAI/bge-small-en-v1.5` for generating high-quality sentence embeddings.

## ⚙️ Setup and Installation

To get the project running locally, follow these steps:

1. **Clone the repository:**

    ```bash
    git clone <https://github.com/Hamza-Elghonemy/AstroNews.git>
    cd AstroNews
    ```

2. **Create and activate a virtual environment:**

    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install the required dependencies:**
    *(Note: A `requirements.txt` file should be created for this)*

    ```bash
    pip install -r requirements.txt
    ```

## ▶️ Usage

Once the setup is complete, you can run the main application script:

```bash
python main.py
```
## 🖼️ UI
<img width="1200" height="520" alt="image" src="https://github.com/user-attachments/assets/e9d7df73-7864-4e11-aafa-3a5344a42bac" />

<img width="1789" height="783" alt="image" src="https://github.com/user-attachments/assets/dde7d605-80ea-4135-b509-b8e233e82527" />


## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
