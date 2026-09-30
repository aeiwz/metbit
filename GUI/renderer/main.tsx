import "@fontsource-variable/inter";
import "./theme.css";
import { render } from "preact";
import { App } from "./App";
import { applyTheme, initialTheme } from "./themePref";

applyTheme(initialTheme()); // before the first paint, so a saved choice never flashes the other palette

render(<App />, document.getElementById("app")!);
