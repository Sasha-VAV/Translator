if __name__ == "__main__":
    import uvicorn

    uvicorn.run("website.main:app", host="0.0.0.0", port=4000, reload=True)
