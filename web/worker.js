// Web Worker for Luminex WASM engine
// Uses ccall to invoke engine functions directly (no stdin/stdout)

let module = null;

function debug(msg) {
  self.postMessage('DBG: ' + msg);
}

self.onmessage = function(e) {
  const msg = e.data;

  if (msg === '__init__') {
    debug('Loading luminex.js...');

    try {
      importScripts('luminex.js');
      debug('luminex.js loaded, typeof createModule=' + typeof createModule);
    } catch(err) {
      self.postMessage('ERROR: Failed to load luminex.js: ' + err.message);
      return;
    }

    if (typeof createModule !== 'function') {
      self.postMessage('ERROR: createModule is not a function. typeof=' + typeof createModule);
      return;
    }

    try {
      createModule({
        print: function(text) {
          self.postMessage(text);
        },
        printErr: function(text) {
          debug('stderr: ' + text);
        },
        noInitialRun: true,
      }).then(function(m) {
        debug('Module created successfully');
        module = m;

        // Check what's available
        debug('typeof ccall=' + typeof m.ccall);
        debug('typeof cwrap=' + typeof m.cwrap);
        debug('typeof _engine_init=' + typeof m._engine_init);
        debug('typeof _process_command=' + typeof m._process_command);
        debug('typeof _main=' + typeof m._main);

        if (typeof m._engine_init === 'undefined') {
          self.postMessage('ERROR: _engine_init not found in module exports');
          return;
        }

        if (typeof m._process_command === 'undefined') {
          self.postMessage('ERROR: _process_command not found in module exports');
          return;
        }

        // Initialize engine
        try {
          debug('Calling engine_init...');
          m.ccall('engine_init', null, [], []);
          debug('engine_init done');
        } catch(err) {
          self.postMessage('ERROR: engine_init failed: ' + err.message);
          return;
        }

        self.postMessage('READY');
      }).catch(function(err) {
        self.postMessage('ERROR: Module init failed: ' + err.message);
      });
    } catch(err) {
      self.postMessage('ERROR: createModule exception: ' + err.message);
    }
    return;
  }

  // Send UCI command via ccall
  if (module) {
    try {
      module.ccall('process_command', null, ['string'], [msg]);
    } catch(err) {
      debug('command error: ' + err.message);
    }
  } else {
    debug('WARNING: command "' + msg + '" received but module not ready');
  }
};

debug('Worker script loaded');
self.postMessage('__loading__');
