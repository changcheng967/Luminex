// Web Worker for Luminex WASM engine
// Handles UCI protocol communication with Emscripten module

let module = null;
let pendingInput = [];
let outputCallback = null;

function initModule() {
  try {
    module = createModule({
      locateFile: (path) => path,
      stdin: () => {
        if (pendingInput.length > 0) {
          return pendingInput.shift();
        }
        return null;
      },
      stdout: (code) => {
        if (outputCallback) {
          outputCallback(String.fromCharCode(code));
        }
      },
      print: (text) => {
        self.postMessage(text);
      },
      printErr: (text) => {
        // Suppress noisy Emscripten stderr
      },
      arguments: [],
      noInitialRun: true,
    });

    module.then((m) => {
      module = m;
      // Start the engine main loop (runs UCI loop)
      m.callMain();
    }).catch((err) => {
      self.postMessage('ERROR: ' + err.message);
    });
  } catch (e) {
    self.postMessage('ERROR: ' + e.message);
  }
}

self.onmessage = (e) => {
  const cmd = e.data;
  if (cmd === '__init__') {
    // Load the WASM module script
    try {
      importScripts('luminex.js');
      initModule();
      // Give Emscripten time to initialize
      setTimeout(() => {
        self.postMessage('READY');
      }, 500);
    } catch (e) {
      self.postMessage('ERROR loading engine: ' + e.message);
    }
    return;
  }
  // Send UCI command to engine via stdin
  pendingInput.push(cmd + '\n');
};

// Kick off initialization
self.postMessage('__loading__');
