// Web Worker for Luminex WASM engine
let module = null;
let crashed = false;

self.onmessage = function(e) {
  const msg = e.data;

  if (msg === '__init__') {
    try {
      importScripts('luminex.js');
    } catch(err) {
      self.postMessage('ERROR: Failed to load luminex.js: ' + err.message);
      return;
    }

    if (typeof createModule !== 'function') {
      self.postMessage('ERROR: createModule is not a function');
      return;
    }

    createModule({
      print: function(text) {
        if (text) self.postMessage(text);
      },
      printErr: function(text) {
        // Silently ignore WASM runtime messages
      },
      noInitialRun: true,
    }).then(function(m) {
      module = m;

      if (typeof m._engine_init === 'undefined') {
        self.postMessage('ERROR: _engine_init not found');
        return;
      }

      try {
        m.ccall('engine_init', null, [], []);
      } catch(err) {
        self.postMessage('ERROR: engine_init failed: ' + err.message);
        return;
      }

      self.postMessage('READY');
    }).catch(function(err) {
      self.postMessage('ERROR: Module init failed: ' + err.message);
    });
    return;
  }

  // Send UCI command
  if (module && !crashed) {
    try {
      module.ccall('process_command', null, ['string'], [msg]);
    } catch(err) {
      crashed = true;
    }
  }
};

self.postMessage('__loading__');
