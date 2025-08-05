def to_tempfile(data, format="pdf"):
    temp_path = f"./temp.{format}"
    with open(temp_path, "wb") as f:
        f.write(data)
    return temp_path
