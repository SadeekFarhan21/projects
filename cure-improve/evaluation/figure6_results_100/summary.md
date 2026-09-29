# Figure-6 Style Metrics Summary

## Config

- Methods: ['cure', 'cure_seq']
- Erased concepts (100): ['Vincent van Gogh', 'Pablo Picasso', 'Rembrandt', 'Andy Warhol', 'Caravaggio', 'Claude Monet', 'Paul Cezanne', 'Edgar Degas', 'Henri Matisse', 'Salvador Dali', 'Pierre-Auguste Renoir', 'Johannes Vermeer', 'Francisco Goya', 'JMW Turner', 'John Constable', 'Gustav Klimt', 'Egon Schiele', 'Paul Gauguin', 'Edvard Munch', 'Wassily Kandinsky', 'Jackson Pollock', 'Mark Rothko', 'Joan Miro', 'Frida Kahlo', 'Diego Rivera', "Georgia O'Keeffe", 'Edward Hopper', 'Jean-Michel Basquiat', 'Keith Haring', 'Roy Lichtenstein', 'Amedeo Modigliani', 'Henri Rousseau', 'William Blake', 'Albrecht Durer', 'Sandro Botticelli', 'Raphael', 'Titian', 'Jan van Eyck', 'Pieter Bruegel the Elder', 'Hieronymus Bosch', 'Gustave Courbet', 'Camille Pissarro', 'Mary Cassatt', 'Winslow Homer', 'John Singer Sargent', 'Claude Lorrain', 'Paul Klee', 'Lucian Freud', 'Francis Bacon', 'Kazimir Malevich', 'Peter Paul Rubens', 'El Greco', 'Arshile Gorky', 'Giorgio de Chirico', 'Umberto Boccioni', 'Georges Braque', 'Fernand Leger', 'Max Ernst', 'Yves Tanguy', 'Rene Magritte', 'Giorgio Morandi', 'Amrita Sher-Gil', 'Nicolas Poussin', 'Jacques-Louis David', 'Eugene Delacroix', 'Jean-Auguste-Dominique Ingres', 'Theodore Gericault', 'Jean-Francois Millet', 'Honore Daumier', 'Alfred Sisley', 'Berthe Morisot', 'Gustave Caillebotte', 'Raoul Dufy', 'Andre Derain', 'Maurice de Vlaminck', 'Georges Seurat', 'Paul Signac', 'M C Escher', 'Katsushika Hokusai', 'Utagawa Hiroshige', 'William Turner', 'Thomas Gainsborough', 'Joshua Reynolds', 'Canaletto', 'Antoine Watteau', 'Jean-Honore Fragonard', 'Giotto', 'Duccio', 'Masaccio', 'Piero della Francesca', 'Tintoretto', 'Paolo Veronese', 'Artemisia Gentileschi', 'Jusepe de Ribera', 'Diego Velazquez', 'Francisco de Zurbaran', 'Annibale Carracci', 'Nicolas de Stael', 'Robert Rauschenberg', 'Cy Twombly']
- Unerased artists (10): ['Norman Rockwell', 'Grant Wood', 'Thomas Cole', 'Albert Bierstadt', 'N C Wyeth', 'Caspar David Friedrich', 'Jean-Baptiste-Camille Corot', 'Artemisia Gentileschi', 'Paul Signac', 'Childe Hassam']
- Checkpoints: [1, 5, 10, 25, 50, 100]
- Seeds: [11, 22, 33]
- Steps: 20, Guidance: 7.5, Size: 384x384

## Method: cure

| # Erased | LPIPSe (↑) | LPIPSu (↓) | CLIP_u (↑) |
|---:|---:|---:|---:|
| 1 | 0.7324 +- 0.1028 | 0.7094 +- 0.0876 | 25.2140 +- 2.8964 |
| 5 | 0.7746 +- 0.0823 | 0.7735 +- 0.0641 | 21.2898 +- 3.3565 |
| 10 | 0.7943 +- 0.0925 | 0.7701 +- 0.0696 | 20.2503 +- 3.3343 |
| 25 | 0.7660 +- 0.0803 | 0.7593 +- 0.0719 | 21.6202 +- 2.8289 |
| 50 | 0.7869 +- 0.0907 | 0.7612 +- 0.0783 | 21.5163 +- 3.0602 |
| 100 | 0.8521 +- 0.1283 | 0.7680 +- 0.0790 | 21.6893 +- 3.0000 |

## Method: cure_seq

| # Erased | LPIPSe (↑) | LPIPSu (↓) | CLIP_u (↑) |
|---:|---:|---:|---:|
| 1 | 0.7324 +- 0.1028 | 0.7094 +- 0.0876 | 25.2140 +- 2.8964 |
| 5 | 0.7587 +- 0.0746 | 0.7375 +- 0.0630 | 22.6821 +- 3.0945 |
| 10 | 0.7678 +- 0.0726 | 0.7397 +- 0.0635 | 22.4827 +- 2.9464 |
| 25 | 0.7570 +- 0.0897 | 0.7350 +- 0.0694 | 22.6564 +- 2.8884 |
| 50 | 0.7577 +- 0.0831 | 0.7470 +- 0.0734 | 22.1578 +- 3.0063 |
| 100 | 0.8540 +- 0.1249 | 0.7939 +- 0.0978 | 21.2503 +- 3.1704 |

## Interpretation

- LPIPSe higher means stronger divergence from baseline on erased artists (stronger removal proxy).
- LPIPSu lower means less collateral change on unerased artists (better preservation).
- CLIP_u higher means better text-image alignment on unerased prompts.
- This reproduces Figure-6-style trend tracking, not a full benchmark.
