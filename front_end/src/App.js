import React from 'react';
import ReactTable, {ReactTableDefaults} from 'react-table';
import logo from './ibm_ix.png';
import './App.css';
import ReactUploadImage from './ImageLoader'
import ImageNetFetcher from './ImageNetFetcher'
import EdgeDetFetcher from './EdgeDetFetcher'
import SegmentDetFetcher from './SegmentDetFetcher'
import ObjectDetFetcher from './ObjectDetFetcher'
import WebCamDetect from './WebCamDetect'

import Container from 'react-bootstrap/Container';
import ToggleButton from 'react-bootstrap/ToggleButton';
import ToggleButtonGroup from 'react-bootstrap/ToggleButtonGroup';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Navbar from 'react-bootstrap/Navbar';
import style from './style';
import ImagenetTable from './ImageNetTable';
import * as cocoSsd from "@tensorflow-models/coco-ssd";

Object.assign(ReactTableDefaults, {
  defaultPageSize: 10,
  minRows: 3,
  showPagination: false,
});

var isBase64 = require('is-base64');

class App extends React.Component {

    constructor(props) {
        super(props);
        this.state = {
            mode: null,
            capture_mode: "image",
            results_single: {
                "object":null,
                "percentage":null
            },
            results_multiple: null,
            file_url: null,
            file: null,
            edge_file_url: null
        };
        this.updateMode = this.updateMode.bind(this);
        this.updateResultSingle = this.updateResultSingle.bind(this);
        this.updateResultMultiple = this.updateResultMultiple.bind(this);
        this.updateImage = this.updateImage.bind(this);
        this.updateEdgeImage = this.updateEdgeImage.bind(this);
        this.onToggleChange = this.onToggleChange.bind(this);
    }

    updateMode (new_mode) {
        this.setState({
            mode: new_mode
        })
    }

    updateResultSingle(response) {
        const new_results_single = {...this.state.results_single};
        new_results_single["object"] = response["object"];
        new_results_single["percentage"] = response["percentage"];

        this.setState({
            results_single: new_results_single
        })
    }

    updateResultMultiple(response) {
        this.setState({
            results_multiple: response
        })
    }

    updateImage (e) {
        this.setState({
            file_url: URL.createObjectURL(e.target.files[0]),
            file: e.target.files[0]
        })
    }

    updateEdgeImage (image) {
        console.log(image);
        this.setState({
            edge_file:image
        })
    }

    onToggleChange(e) {
        console.log(e);
        this.setState({
            capture_mode: e
        })
    }

    render() {
        return (
            <div className="App">


                {/* NAVBAR */}
                <Navbar sticky="top" className="navBar" className="justify-content-between">
                    <div>
                        <img src={logo} className="image"/>
                        TorchVision Project
                    </div>
                    <ToggleButtonGroup
                        type="radio"
                        name="videoimage"
                        onChange={this.onToggleChange}
                    >
                        <ToggleButton checked={true} value={'video'} variant="light">Video</ToggleButton>
                        <ToggleButton name={'image'} value={'image'} variant="light">Image</ToggleButton>
                    </ToggleButtonGroup>
                </Navbar>


                {/* MAIN SECTION */}
                <Container fluid={true} className="mainContainer">
                    <Row className="mainRow">

                        {/* SIDE BAR */}
                        <Col md={3} lg={3} className="sideColVideo">
                            {
                                this.state.capture_mode === "image" ?
                                <div>
                                    <Row className="imageNetButton">
                                        <ImageNetFetcher

                                            file={this.state.file}
                                            updateResultMultiple={this.updateResultMultiple}
                                            updateMode={this.updateMode}

                                        />

                                    </Row>
                                    <Row className="imageNetButton">
                                        <ObjectDetFetcher
                                            file={this.state.file}
                                            updateEdgeImage={this.updateEdgeImage}
                                            updateMode={this.updateMode}

                                        />
                                    </Row>
                                    <Row className="imageNetButton">
                                        <EdgeDetFetcher
                                            file={this.state.file}
                                            updateEdgeImage={this.updateEdgeImage}
                                            updateMode={this.updateMode}

                                        />

                                    </Row>

                                    <Row className="imageNetButton">
                                        <SegmentDetFetcher
                                            file={this.state.file}
                                            updateEdgeImage={this.updateEdgeImage}
                                            updateMode={this.updateMode}

                                        />
                                    </Row>
                                </div>
                                :
                                <div>
                                    <Row className="imageNetButton">
                                        <div>
                                        <h4>Live Object Detection</h4>
                                        <p>Use pretrained object detection models to detect objects with corresponding labels in live stream in image</p>
                                        </div>
                                    </Row>
                                </div>
                                }

                        </Col>

                        {/* MAIN COLUMN */}
                        <Col md={9} lg={9} className="mainCol">

                            {
                                this.state.capture_mode === "video" ?
                                <WebCamDetect/>
                                :
                            <div>
                                <div className="uploadDiv">



                                    <ReactUploadImage

                                        updateResultSingle={this.updateResultSingle}
                                        updateImage={this.updateImage}

                                        file_url={this.state.file_url}
                                        file={this.state.file}
                                    />



                                </div>

                                <div className="resultsDiv">

                                    {
                                        this.state.mode == "imagenet" ?
                                        <div className="imageNetTables">
                                            <div className="resnetTable">
                                                <h4>Alexnet</h4>

                                                <ImagenetTable

                                                    results_multiple={this.state.results_multiple.alexnet}

                                                />
                                            </div>
                                            <div className="resnetTable">
                                                <h4>ResNet101</h4>

                                                <ImagenetTable

                                                    results_multiple={this.state.results_multiple.resnet}

                                                />
                                            </div>
                                            <div className="resnetTable">
                                                <h4>MobileNet</h4>

                                                <ImagenetTable

                                                    results_multiple={this.state.results_multiple.mobilenet}

                                                />
                                            </div>
                                        </div>

                                        :
                                        null
                                    }
                                    {
                                        this.state.mode == "edge" ?
                                        <div className="previewBackground">
                                            <img src={`data:image/jpeg;base64,${this.state.edge_file}`} className="previewImage"/>
                                        </div>:
                                        null
                                    }
                                    {
                                        this.state.mode == "object" ?
                                        <div className="previewBackground">
                                            <img src={`data:image/jpeg;base64,${this.state.edge_file}`} className="previewImage"/>
                                        </div>:
                                        null
                                    }

                                </div>
                            </div>
                            }
                        </Col>

                    </Row>
                </Container>
            </div>
        );
    }

}

export default App;
