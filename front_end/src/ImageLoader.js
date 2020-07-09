import './App.css';

import React from 'react'
const axios = require("axios");


class ReactUploadImage extends React.Component {

    constructor(props) {
        super(props);
        this.onFormSubmit = this.onFormSubmit.bind(this);
        this.onChange = this.onChange.bind(this);
        this.postImageData = this.postImageData.bind(this);
        this.refreshResultData = this.refreshResultData.bind(this);
    }

    onFormSubmit(e){
        e.preventDefault();
        const formData = new FormData();
        formData.append('myImage',this.state.file);
        this.postImageData(formData).then(this.refreshResultData);
    }

    postImageData (formData) {
        const config = {
            headers: {
                'content-type': 'multipart/form-data'
            }
        };

        return axios.post('http://localhost:7050/inference',formData,config)
            .then(res => res.data);
    }

    refreshResultData = (res) => {
        this.props.updateResultSingle(res)
    }

    onChange(e) {
        this.props.updateImage(e)
    }

    render() {
        return (

            <div>
            {
                this.props.file != null ?
                <div className="previewBackground">
                    <img src={this.props.file_url} className="previewImage"/>
                </div>:
                null
            }

            <input type="file" name="myImage" onChange= {this.onChange} />

            </div>
        )
    }
}

export default ReactUploadImage