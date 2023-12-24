from os import path, walk, name as os_name
from uuid import uuid4
import streamlit as st
import random
import tkinter as tk
from tkinter import filedialog
from AI.engine import AIEngine
import chromadb
from settings import Settings
from checksumdir import dirhash
import subprocess
from pathlib import Path
from Database import crud, models, schemas
from Database.database import SessionLocal, engine as db_engine

root = tk.Tk()
root.withdraw()
root.wm_attributes("-topmost", 1)

settings = Settings()

client = chromadb.PersistentClient(path=path.join(settings.ROOT_DIR, "vdb"))

ai_engine = AIEngine()

models.Base.metadata.create_all(bind=db_engine)


def update_stat():
    collections = client.list_collections()
    image_collections = [c.name for c in collections if "_image" in c.name]
    text_collections = [c.name for c in collections if "_text" in c.name]
    st.session_state.collections = {
        "image": image_collections,
        "text": text_collections,
    }
    print("COLLECTIONS IMG", image_collections)
    print("COLLECTIONS TXT", text_collections)


def list_image_paths(directory):
    image_extensions = [".png", ".jpg", ".jpeg", ".ppm", ".gif", ".tiff", ".bmp"]
    image_paths = []

    try:
        for root, _, files in walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(path.normpath(path.join(root, file)))

    except OSError as e:
        print(f"Error reading directory '{directory}': {e}")

    return image_paths


def open_file_explorer(file_path):
    if not path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    try:
        if os_name == "nt":
            subprocess.Popen(["explorer", "/select,", file_path], shell=True)
        elif os_name == "posix":
            subprocess.Popen(["open", "-R", file_path])
        elif os_name == "posix":
            subprocess.Popen(["nautilus", "--select", file_path])
        else:
            print("Unsupported operating system.")
    except Exception as e:
        print(f"Error: {e}")


def get_collection_dict(collection_list: list[str]):
    collection_dict = {}
    db = SessionLocal()
    for collection in collection_list:
        print("COLLECTION", collection)
        path = crud.get_index_path(db=db, index_id=collection.split("_")[0])
        print("PATH", path)
        collection_dict[path.indexPath] = collection
    db.close()
    return collection_dict


def createAndOperation(path: str):
    img_path_list = list_image_paths(path)
    text_collection = client.get_or_create_collection(name=f"{dirhash(path)}_text")
    image_collection = client.get_or_create_collection(name=f"{dirhash(path)}_image")
    print("I P L", img_path_list)
    caption_list = [ai_engine.generate_caption(img_path) for img_path in img_path_list]
    caption_emb_list = ai_engine.generate_text_embedding(caption_list)
    image_emb_list = [
        ai_engine.generate_image_embedding(img_path) for img_path in img_path_list
    ]
    metadatalist = [{"path": img_path} for img_path in img_path_list]
    text_collection.upsert(
        ids=[str(hash(img_path)) for img_path in img_path_list],
        embeddings=caption_emb_list,
        metadatas=metadatalist,  # type: ignore
        documents=caption_list,
    )
    image_collection.upsert(
        ids=[str(hash(img_path)) for img_path in img_path_list],
        embeddings=image_emb_list,
        metadatas=metadatalist,  # type: ignore
        documents=caption_list,
    )
    db = SessionLocal()
    crud.create_index(
        db=db, index=schemas.IndexCreate(indexId=dirhash(path), indexPath=path)
    )
    db.close()
    update_stat()


if "img_paths" not in st.session_state:
    st.session_state.img_paths = []

update_stat()

st.set_page_config(page_title="File Sense", page_icon="🔍", layout="wide")

st.title("File Sense (PoC) 🔍🖼️", anchor="center")

clicked = st.button("Select Folder for Indexing")
if clicked:
    dirname = filedialog.askdirectory(master=root, title="Select a folder to Index")  # type: ignore
    with st.spinner("Indexing in Progress"):
        createAndOperation(dirname)
    st.success("Indexing Completed")
    st.rerun()

textToImageTab, imageToImageTab = st.tabs(
    ["Search Image by Text", "Search Image by Image"]
)

with textToImageTab:
    with st.container():
        st.header("Search Image by Text")
        with st.form("Search_Text_Form"):
            row1 = st.columns([1, 3, 1, 1])
            collection_dict = get_collection_dict(st.session_state.collections["text"])
            select_collection = ""
            if len(collection_dict) != 0:
                select_collection = row1[0].selectbox(  # type: ignore
                    "Select Collection", tuple(list(collection_dict.keys()))
                )
                select_collection = collection_dict[select_collection]
            text = row1[1].text_input("Enter Text", placeholder="Enter Text to Search")
            no_of_results = row1[2].number_input(
                "No of Results", min_value=1, max_value=10, value=3
            )
            submitted = row1[3].form_submit_button("Search :mag:")
            if submitted:
                st.session_state.img_paths = []
                if text is None or text == "":
                    st.toast("Please Enter Text to Search", icon="🚨")
                else:
                    print("TEXT", text)
                    print("COLLECTION", select_collection)
                    print("NO OF RESULTS", no_of_results)
                    text_collection = client.get_collection(select_collection)
                    print("TEXT COLLECTION", text_collection)
                    text_emb = ai_engine.generate_text_embedding([text])
                    results = text_collection.query(
                        query_embeddings=text_emb,
                        n_results=no_of_results,  # type: ignore
                    )
                    print("RESULTS", results)
                    for idx, result in enumerate(results["metadatas"]):  # type: ignore
                        for value in result:
                            image_path = path.relpath(value["path"], Path.cwd())  # type: ignore
                            st.session_state.img_paths.append(image_path)
        st.button(
            "Clear",
            on_click=lambda: st.session_state.img_paths.clear(),
            type="primary",
            key=uuid4().hex,
        )
        with st.container():
            if len(st.session_state.img_paths) > 0:
                if len(st.session_state.img_paths) <= 5:
                    img_rows = st.columns(
                        [1 for _ in range(len(st.session_state.img_paths))]
                    )
                    for idx, img_path in enumerate(st.session_state.img_paths):
                        img_rows[idx].image(img_path)
                        img_rows[idx].write(img_path)
                else:
                    img_rows = st.columns([1 for _ in range(5)])
                    for idx, img_path in enumerate(st.session_state.img_paths[:5]):
                        img_rows[idx].image(img_path)
                        img_rows[idx].write(img_path)
                    img_rows = st.columns([1 for _ in range(5)])
                    for idx, img_path in enumerate(st.session_state.img_paths[5:]):
                        img_rows[idx].image(img_path)
                        img_rows[idx].write(img_path)
with imageToImageTab:
    with st.container():
        st.header("Search Image by Image")
        with st.form("Search_Image_Form"):
            row1 = st.columns([1, 3, 1, 1])
            collection_dict = get_collection_dict(st.session_state.collections["image"])
            select_collection = ""
            if len(collection_dict) != 0:
                select_collection = row1[0].selectbox(  # type: ignore
                    "Select Collection", tuple(list(collection_dict.keys()))
                )
                select_collection = collection_dict[select_collection]
            upload_image = row1[1].file_uploader(
                label="Upload Reference Image",
                type=["png", "jpg", "jpeg", "ppm", "gif", "tiff", "bmp"],
            )
            no_of_results = row1[2].number_input(
                "No of Results", min_value=1, max_value=10, value=3
            )
            submitted = row1[3].form_submit_button("Search :mag:")
            if submitted:
                st.session_state.img_paths = []
                if upload_image is None:
                    st.toast("Please Upload Image to Search", icon="🚨")
                else:
                    print("IMAGE", upload_image)
                    print("COLLECTION", select_collection)
                    print("NO OF RESULTS", no_of_results)
                    image_collection = client.get_collection(select_collection)
                    print("IMAGE COLLECTION", image_collection)
                    image_emb = ai_engine.generate_image_embedding(upload_image)  # type: ignore
                    print("IMAGE EMB", image_emb)
                    results = image_collection.query(
                        query_embeddings=image_emb,
                        n_results=no_of_results,  # type: ignore
                    )
                    print("RESULTS", results)
                    for idx, result in enumerate(results["metadatas"]):  # type: ignore
                        for value in result:
                            image_path = path.relpath(value["path"], Path.cwd())  # type: ignore
                            st.session_state.img_paths.append(image_path)
        st.button(
            "Clear",
            on_click=lambda: st.session_state.img_paths.clear(),
            type="primary",
            key=uuid4().hex,
        )
    with st.container():
        if len(st.session_state.img_paths) > 0:
            if len(st.session_state.img_paths) <= 5:
                img_rows = st.columns(
                    [1 for _ in range(len(st.session_state.img_paths))]
                )
                for idx, img_path in enumerate(st.session_state.img_paths):
                    img_rows[idx].image(img_path)
                    img_rows[idx].caption(img_path)
            else:
                img_rows = st.columns([1 for _ in range(5)])
                for idx, img_path in enumerate(st.session_state.img_paths[:5]):
                    img_rows[idx].image(img_path)
                    img_rows[idx].caption(img_path)
                img_rows = st.columns([1 for _ in range(5)])
                for idx, img_path in enumerate(st.session_state.img_paths[5:]):
                    img_rows[idx].image(img_path)
                    img_rows[idx].caption(img_path)
