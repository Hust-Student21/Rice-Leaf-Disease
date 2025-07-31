import "../style/loader.css";

const Loader = (props) => { // the function name loader with the input variable = progress
  return (
    // to open the spinning icon
    <div className="wrapper"> 
    {/* spin the spinning icon */}
      <div className="spinner"></div>
      <p>{props.children}</p>
    </div>
  );
};

export default Loader;
