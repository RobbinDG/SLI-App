function request(endpoint, data, onSuccess, onError) {
    let xhr = new XMLHttpRequest();
    try {
        xhr.open('POST', endpoint, true);
        // xhr.setRequestHeader('Content-Type', 'application/json;charset=UTF-8');
        xhr.onload = function () {
            if (xhr.status === 200) {
                onSuccess(JSON.parse(xhr.response));
            } else {
                onError();
            }
        };
        xhr.send(data);
    } catch (e) {
        onError();
    }
}