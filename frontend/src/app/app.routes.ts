import { Routes } from '@angular/router';
import { WelcomeComponent } from './features/welcome/welcome.component';
import { AboutComponent } from './features/about/about.component';
import { LabComponent } from './features/laboratory/lab.component';

export const routes: Routes = [
  { path: '', component: WelcomeComponent, title: 'Qapha' },
  { path: 'lab', component: LabComponent, title: 'Qapha Lab' },
  { path: 'how-it-works', component: AboutComponent, title: 'How Qapha works' },
  { path: '**', redirectTo: '' },
];
