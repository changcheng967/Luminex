// Web Worker for Luminex WASM engine
// Uses ccall to invoke engine functions directly (no stdin/stdout)

let module = null;
let stdoutBuffer = '';

self.onmessage = function(e) {
  const msg = e.data;

  if (msg === '__init__') {
    try {
      importScripts('luminex.js');
    } catch(err) {
      self.postMessage('ERROR: Failed to load luminex.js: ' + err.message);
      return;
    }

    try {
      createModule({
        print: function(text) {
          self.postMessage(text);
        },
        printErr: function(text) {
          // Suppress noisy Emscripten internals
        },
        noInitialRun: true,
      }).then(function(m) {
        module = m;
        // Initialize engine (magic bitboards, TT, etc.)
        m.ccall('engine_init', null, [], []);
        self.postMessage('READY');
      }).catch(function(err) {
        self.postMessage('ERROR: Module init failed: ' + err.message);
      });
    } catch(err) {
      self.postMessage('ERROR: ' + err.message);
    }
    return;
  }

  // Send UCI command via ccall
  if (module && msg !== '__loading__') {
    try {
      module.ccall('process_command', null, ['string'], [msg]);
    } catch(err) {
      self.postMessage('ERROR: command failed: ' + err.message);
    }
  }
};

self.postMessage('__loading__');
