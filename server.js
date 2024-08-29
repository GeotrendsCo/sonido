const express = require('express');
const path = require('path');
const fs = require('fs');

const app = express();
const port = 3000;

// Configurar la carpeta 'public' como estática
app.use(express.static(path.join(__dirname, 'public')));

// Manejar rutas adicionales si es necesario
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.get('/circle_packing_deep.html', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'circle_packing_deep.html'));
});

app.get('/visual.html', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'visual.html'));
});

// Endpoint para circle_packing_dataDeep.json
app.get('/circle-packing-deep-data', (req, res) => {
  const dataPath = path.join(__dirname, 'public/circle_packing_dataDeep.json');
  fs.readFile(dataPath, 'utf8', (err, data) => {
      if (err) {
          console.error('Error al leer el archivo JSON:', err);
          res.status(500).send('Error al leer el archivo JSON');
          return;
      }
      try {
          const jsonData = JSON.parse(data);
          res.json(jsonData);
      } catch (parseErr) {
          console.error('Error al parsear el archivo JSON:', parseErr);
          res.status(500).send('Error al parsear el archivo JSON');
      }
  });
});

app.get('/force-directed-tree.html', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'force-directed-tree.html'));
});

// Endpoint para force_directed_tree_data.json
app.get('/force-directed-tree-data', (req, res) => {
  const dataPath = path.join(__dirname, 'public/force_directed_tree_data.json');
  fs.readFile(dataPath, 'utf8', (err, data) => {
      if (err) {
          console.error('Error al leer el archivo JSON:', err);
          res.status(500).send('Error al leer el archivo JSON');
          return;
      }
      try {
          const jsonData = JSON.parse(data);
          res.json(jsonData);
      } catch (parseErr) {
          console.error('Error al parsear el archivo JSON:', parseErr);
          res.status(500).send('Error al parsear el archivo JSON');
      }
  });
});

app.get('/radial_tree.html', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'radial_tree.html'));
});


// Endpoint para radial_tree_data.json
app.get('/radial-tree-data', (req, res) => {
  const dataPath = path.join(__dirname, 'public/radial_tree_data.json');
  fs.readFile(dataPath, 'utf8', (err, data) => {
      if (err) {
          console.error('Error al leer el archivo JSON:', err);
          res.status(500).send('Error al leer el archivo JSON');
          return;
      }
      try {
          const jsonData = JSON.parse(data);
          res.json(jsonData);
      } catch (parseErr) {
          console.error('Error al parsear el archivo JSON:', parseErr);
          res.status(500).send('Error al parsear el archivo JSON');
      }
  });
});

app.get('/about', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'about.html'));
});

app.get('/contact', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'contact.html'));
});

app.listen(port, () => {
  console.log(`Servidor corriendo en http://localhost:${port}`);
});
