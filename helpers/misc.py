def format_response(response: dict) -> str:
    answer = response.get("answer", "")
    if not answer:
        return ""
    contexts = response.get("context", {})

    for idx, context in enumerate(contexts):
        doc_name = context.metadata["doc_name"]
        page_number = context.metadata["page_num"]
        # Append the metadata to the answer string
        answer += f"\n\n - Source ({idx+1}) - Document: {doc_name}, Page number: {page_number}"
    return answer
