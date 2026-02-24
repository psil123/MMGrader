import requests, cv2
def crop_poly(image,points,width,height):
    img = cv2.resize(image, (width,height))
    points = np.array(points, dtype=np.float32)
    points[:, 0] = points[:, 0] * width/100   # x-coordinates
    points[:, 1] = points[:, 1] * height/100
    points = points.astype(np.int32)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    res = cv2.bitwise_and(img, img, mask=mask)
    rect = cv2.boundingRect(points)
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    return cropped

def vllm_call(model_url, system_prompt, user_prompt):
    headers = {
                "Content-Type": "application/json",
                # "Authorization": "Bearer bharatgen-secret-token-123" 
                }
    payload = {   
        # "model": "param-17B SFT S1",
        "temperature": 0.4,
        "max_length": 2048,
        "chat_template_kwargs": {
            "enable_thinking": True
        },
        "messages": [
            {
            "role": "system",
            "content": system_prompt
            },
            {
            "role": "user",
            "content": user_prompt
            }
        ]
        }
    resp = requests.post(model_url, headers=headers, json=payload, verify=False)
    try:
        resp=resp.json()
        resp=resp['choices'][0]['message']['content']
    except Exception as e:
        print("Exception : ",e)
        return None
    return resp.strip()