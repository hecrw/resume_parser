# import requests

# url = "http://127.0.0.1:8000/parse_resume/"
# file_path = "test.pdf"

# with open(file_path, "rb") as f:
#     files = {"file": (file_path, f, "application/pdf")}
#     response = requests.post(url, files=files)

# print(response.json())

# from pathlib import Path
# from paddleocr import PPStructureV3

# input_file = "./test.pdf"
# output_path = Path("./output")

# pipeline = PPStructureV3(enable_mkldnn=False)
# output = pipeline.predict(input=input_file)

# markdown_list = []
# markdown_images = []

# for res in output:
#     md_info = res.markdown
#     markdown_list.append(md_info)
#     markdown_images.append(md_info.get("markdown_images", {}))

# markdown_texts = pipeline.concatenate_markdown_pages(markdown_list)

# mkd_file_path = output_path / f"{Path(input_file).stem}.md"
# mkd_file_path.parent.mkdir(parents=True, exist_ok=True)

# with open(mkd_file_path, "w", encoding="utf-8") as f:
#     f.write(markdown_texts.get("markdown_texts", ""))

# for item in markdown_images:
#     if item:
#         for path, image in item.items():
#             file_path = output_path / path
#             file_path.parent.mkdir(parents=True, exist_ok=True)
#             image.save(file_path) 