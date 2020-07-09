import './App.css';

import React from 'react'
const axios = require("axios");

class ImageNetFetcher extends React.Component {

    constructor(props) {
        super(props);
        this.onClick = this.onClick.bind(this);
        this.postImageData = this.postImageData.bind(this);
        this.refreshResultData = this.refreshResultData.bind(this);
    }

    postImageData () {
        const formData = new FormData();
        formData.append("myImage", this.props.file);

        const config = {
            headers: {
                'content-type': 'multipart/form-data'
            }
        };
        return axios.post('http://localhost:7050/inference',formData,config)
            .then(res => res.data);
    }

    refreshResultData = (res) => {
        this.props.updateResultMultiple(res);
        this.props.updateMode("imagenet");
    }

    onClick(e) {
        if (this.props.file != null) {
            this.postImageData()
            .then(this.refreshResultData)
        };
    }

    render() {
        return (

            <div onClick={this.onClick}>
                <h4>ImageNet CNNs</h4>
                <p>Use models pretrained on the imagenet dataset to predict a single object represented by the image</p>
            </div>
        )
    }
}

export default ImageNetFetcher;