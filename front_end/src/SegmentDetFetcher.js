import './App.css';

import React from 'react'
const axios = require("axios");

class SegmentDetFetcher extends React.Component {

    constructor(props) {
        super(props);
        this.onClick = this.onClick.bind(this);
        this.postImageData = this.postImageData.bind(this);
    }

    postImageData () {
        const formData = new FormData();
        formData.append("myImage", this.props.file);

        const config = {
            headers: {
                'content-type': 'multipart/form-data'
            }
        };
        return axios.post('http://localhost:7050/segmentdetection', formData,config,
            { responseType: 'arraybuffer' })
            .then(res => res.data);
    }

    onClick(e) {
        if (this.props.file != null) {
            this.postImageData()
                .then(this.props.updateEdgeImage)
        };
        this.props.updateMode("object");
    }

    render() {
        return (

            <div onClick={this.onClick}>
                <h4>Segment Detection</h4>
                <p>Use pretrained object detection models to detect objects with corresponding labels in image</p>
            </div>
        )
    }
}

export default SegmentDetFetcher;