from flask import Flask, render_template, request, jsonify
from app_sa_llm_trans_my import preprocess_user_query, get_relevant_chunks, generate_response, postprocess_answer, format_llm_output

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    # Step 1: Preprocess user query
    query_data = preprocess_user_query(user_message)
    user_query_eng = query_data["user_query_eng"]
    user_lang = query_data["user_original_query_lang"]

    # Step 2: Retrieve context
    relevant_chunks = get_relevant_chunks(user_query_eng)

    # Step 3: Generate English response
    final_response_eng = generate_response(user_query_eng, relevant_chunks)

    # Step 4: Postprocess answer into original language (if needed)
    final_response = postprocess_answer(final_response_eng, user_lang)
    formatted_response = format_llm_output(final_response)

    return jsonify({"response": formatted_response})

if __name__ == "__main__":
    app.run(debug=True)
