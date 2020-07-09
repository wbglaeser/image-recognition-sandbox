import ReactTable, {ReactTableDefaults} from 'react-table';
import React from 'react';
import 'react-table/react-table.css'
import Button from 'react-bootstrap/Button';


class ImagenetTable extends React.Component {

    // Call constructor
    constructor(props) {
        super(props);
    }

    // Render
    render() {

        // Define columns
        const columns = [
        {
            Header: 'Object',
            Cell: row => <div style={{ textAlign: "start" }}>{row.value}</div>,
            accessor: 'object', // String-based value accessors!
            width: 150,
        },
        {
            Header: 'Percentage',
            accessor: 'percentage', // String-based value accessors!
            width: 150,
        }];

        return <ReactTable
                    data={this.props.results_multiple}
                    columns={columns}
                />

    }
}

export default ImagenetTable;