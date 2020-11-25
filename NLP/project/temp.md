@app.route('arch')
def search():
    key_words = request.args.get('key_words')
    offset = request.args.get("offset")
    page_size = request.args.get("page_size")

    try:
        offset = int(offset)
        page_size = int(page_size)
    except:
        offset = 0
        page_size = 10
    index_name = "zhihu"


    start_time = datetime.now()
    response = client.search(
        index= index_name,
        body={
            "query":{
                "multi_match":{
                    "query": key_words,
                    "fields": ["title", "content", "topics"]
                }
            },
            "from": offset*page_size,
            "size": page_size,
            "highlight": {
                "pre_tags": ['<span class="keyWord">'],
                "post_tags": ['</span>'],
                "fields": {
                    "title": {},
                    "content": {}
                }
            }
        }
    )

    end_time = datetime.now()
    last_seconds = (end_time-start_time).total_seconds()
    total_nums = response["hits"]["total"]
    hit_list = []
    for hit in response["hits"]["hits"]:
        print(hit)
        from collections import defaultdict
        hit_dict = defaultdict(str)
        if "highlight" not in hit:
            hit["highlight"] = {}
        if "title" in hit["highlight"]:
            hit_dict["title"] = "".join(hit["highlight"]["title"])
        else:
            hit_dict["title"] = hit["_source"]["title"]

        if "content" in hit["highlight"]:
            hit_dict["content"] = "".join(hit["highlight"]["content"])
        else:
            hit_dict["content"] = hit["_source"]["content"]

        if "create_date" in hit_dict:
            hit_dict["create_date"] = hit["_source"]["create_date"]
        if "publish_time" in hit["_source"]:
            hit_dict["create_date"] = hit["_source"]["publish_time"]
        hit_dict["url"] = hit["_source"]["url"]
        hit_dict["topics"] = hit["_source"]["topics"]
        hit_dict["answer_num"] = hit["_source"]["answer_num"]
        hit_dict["watch_user_num"] = hit["_source"]["watch_user_num"]
        hit_dict["click_num"] = hit["_source"]["click_num"]
        hit_dict["score"] = hit["_score"]

        hit_list.append(hit_dict)

    return resp(0, {"all_hits":hit_list,
            "key_words":key_words,
            "total_nums":total_nums,
            "last_seconds":last_seconds,})
