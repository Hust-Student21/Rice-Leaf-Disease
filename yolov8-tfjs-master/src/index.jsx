import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./style/index.css";

const root = createRoot(document.getElementById("root")); // define root variable
root.render(// render app with id = root
  <React.StrictMode> 
    <App />
  </React.StrictMode>
);
// App.js connect with index.html through index.js
