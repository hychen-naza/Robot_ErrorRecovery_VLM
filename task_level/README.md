# gpt_ask

install with ```pip install -r requirements.txt```

Main code in gpt_call.py.

Run with `python gpt_call.py <target directory>`, where target directory should at least contain a `query.json` file.

### query.json

See experiments directory for example.

The json should has a `system_msg`, and a `query`. The `query` list is a list of questions to ask GPT. Each question in the `query` list has two parts. The first part is the text prompt, and the second part is a list of image paths. If no image is needed, you can enter `null` and only text will be provided to GPT.

Example: 

```
{
    "system_msg": "The system message",
    "query": [
        ["The first element of each query is the text", ["/path/to/image/1.jpg","/path/to/image/2.jpg"]],
        ["Can enter multiple rounds of query into the list", [/path/to/image/3.jpg]],
        ["Can enter null if image is not needed", null]
    ]
}
```