# faster-image-match
Search images faster using neural net features and elasticsearch
Any image can be represented as a feature vector, where each vector is the final layer output or the output of a fully connected layer of a neral net.
These feature vectors can be indexed to an elasticsearch to enable faster image match with a search initiated with an input image.
## Create new elasticsearch index
To create a fresh new index, run

``` python create_index.py --index <name-of-index> --confid <path_to_json_file> ```

## Add or match new image in search
Running ``` python image_search.py ``` will activate a flask app at ```http://localhost:4323```

### To add image 

send post request to ```http://localhost:4323/add_image``` as
```json
{"image": <image_file>,
"tag": "name_tag_for_image",
"name_of_index": "index_name_in_elasticsearch"
}
```

### To search image

send post request to ```http://localhost:4323/match_image``` as
```json
{"image": <image_file>,
"name_of_index": "index_name_in_elasticsearch"
}
```

### Postman collection link:
[https://www.getpostman.com/collections/38a22c56bd63a0944d54](https://www.getpostman.com/collections/38a22c56bd63a0944d54)
