import { Component } from '@angular/core';
import { RouterLink } from '@angular/router';

@Component({
  selector: 'qapha-about',
  imports: [RouterLink],
  templateUrl: './about.component.html',
})
export class AboutComponent {
  readonly sections = [
    ['goal', 'The ultimate goal'],
    ['strategy', 'How the experiments lead there'],
    ['why-ball', 'Why a bouncing ball'],
    ['loop', 'The predict-and-check loop'],
    ['meta', 'How Meta AI is used'],
    ['models', 'The models compared'],
    ['uncertainty', 'Knowing how unsure it is'],
    ['floor', 'Learning the floor'],
    ['discipline', 'Keeping the science honest'],
    ['results', 'Results'],
    ['failures', 'What failed and why'],
    ['code', 'Code structure'],
    ['limits', 'Limitations'],
    ['next', 'What comes next'],
    ['lab', 'How to read the lab'],
  ];

  go(id: string): void {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
  }
}
