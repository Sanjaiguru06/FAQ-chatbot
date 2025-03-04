
import gradio as gr
from huggingface_hub import InferenceClient
from urllib.parse import quote
import time

# Initialize the LLM
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

def fetch_articles(query):
    """Fetch relevant articles from the web."""
    search_query = f"{query} site:geeksforgeeks.org OR site:medium.com OR site:stackexchange.com"
    search_url = f"https://www.google.com/search?q={quote(search_query)}"
    return f"[Search related articles here]({search_url})"

def fetch_youtube_links(query):
    """Generate a YouTube search link for related videos."""
    yt_search_url = f"https://www.youtube.com/results?search_query={quote(query)}"
    return f"[Watch related videos on YouTube]({yt_search_url})"

def respond(
    message,
    history=None,
    system_message="A helpful AI assistant providing detailed answers with resources.",
    max_tokens=512,
    temperature=0.7,
    top_p=0.95,
):
    """Main chatbot function"""
    
    # Debugging: Print received message
    print(f"Received message: {message}")  # ✅ Check if message is received from examples

    if history is None:
        history = []

    messages = [{"role": "system", "content": system_message}]
    
    for val in history:
        if val[0]: messages.append({"role": "user", "content": val[0]})
        if val[1]: messages.append({"role": "assistant", "content": val[1]})
    
    messages.append({"role": "user", "content": message})

    try:
        response = ""
        for msg in client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
        ):
            token = msg.choices[0].delta.content
            response += token

        # Fetch external resources
        article_links = fetch_articles(message)
        youtube_links = fetch_youtube_links(message)

        # Final response
        final_output = f"**Response:**\n{response}\n\n📖 **Useful Articles:** {article_links}\n\n📺 **YouTube Resources:** {youtube_links}"
        
        print("✅ Successfully generated response!")  # Debugging print
        return final_output  

    except Exception as e:
        print("❌ Error:", e)
        return "❌ Error: Unable to generate a response."

# Define the chatbot interface
chat = gr.ChatInterface(
    respond,
    examples=[
        ["How do I reset my password?"],
        ["What is cloud computing?"],
        ["How to learn Python?"],
        ["What are the benefits of AI?"]
    ],
    additional_inputs=[
        gr.Textbox(value="A helpful AI assistant providing detailed answers with resources.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)"),
    ],
    submit_btn="Ask AI"
)

# Debugging: Ensure example queries work
def test_example_input():
    test_query = "How do I reset my password?"
    print("🔄 Testing example input...")
    response = respond(test_query, history=[])
    print("📝 Response:", response)

if __name__ == "__main__":
    test_example_input()  # ✅ Run a test before launching
    chat.launch(share=True)

