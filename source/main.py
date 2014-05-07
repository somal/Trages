from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.uix.button import Button
from time import sleep
import recognizer
from kivy.uix.accordion import Accordion,AccordionItem

def go_exit(x):
        Window.close()
        exit()

def go_children(x):
        Window.close()
        recognizer.main()
        #Window.size=(800,600)
        

def go_game(x):
        Window.close()
        import game
        game.main()
        
def main_menu():
        #Window.fullscreen=True
        Window.clearcolor=(1,140.0/255,15.0/255,0)
        Menu=BoxLayout(orientation='vertical')
        title = Label(
            text='Trages',
            markup=True,
            bold=True,
            color=(79.0/255,15.0/255,204.0/255,0),
            font_name='RAVIE.ttf',
            font_size='100dp',
            y=Window.height/2-25,
            x=-Window.width/2+100,
            size_hint=(1,0.3)
        )
        Menu.add_widget(title)
        
        root=Accordion(orientation='vertical')

        ButtonChildren=Button(text='Press here for children education',size_hint=(1,0.3))
        ButtonChildren.bind(on_press=go_children)

        s1='This version of software\n is a new method of \nteaching children.\n It allows one to make \nlearning process \nmore interactive and \nsimple due to gaming form.'
        LabelChildren=Label(text=s1, font_name='RAVIE.ttf',font_size='20dp',max_lines=4,shorten=True,color=(113.0/255,17.0/255,150.0/255,1))       
        BoxLayoutChildren=BoxLayout(orientation='horizontal')
        BoxLayoutChildren2=BoxLayout(orientation='vertical')
        BoxLayoutChildren2.add_widget(LabelChildren)
        BoxLayoutChildren2.add_widget(ButtonChildren)
        ImageChildren=Image(source='childeduc.bmp')
        BoxLayoutChildren.add_widget(ImageChildren)
        BoxLayoutChildren.add_widget(BoxLayoutChildren2)
        
        children=AccordionItem(title='Children Education')      
        children.add_widget(BoxLayoutChildren)
        
        ###
        ButtonGame=Button(text='Press here for testing',size_hint=(1,.3))
        ButtonGame.bind(on_press=go_game)
        s2='This version of software\n is a new method of \ntesting children.\n It allows one to make \ntesting process \nmore interactive and \nsimple due to gaming form.'
        LabelGame=Label(text=s2, font_name='RAVIE.ttf',font_size='20dp',max_lines=4,shorten=True,color=(113.0/255,17.0/255,150.0/255,1))       
        BoxLayoutGame=BoxLayout(orientation='horizontal')
        BoxLayoutGame2=BoxLayout(orientation='vertical')
        BoxLayoutGame2.add_widget(LabelGame)
        BoxLayoutGame2.add_widget(ButtonGame)
        ImageGame=Image(source='forgame.bmp')
        BoxLayoutGame.add_widget(ImageGame)
        BoxLayoutGame.add_widget(BoxLayoutGame2)
                                 
        game=AccordionItem(title='Game!')
        game.add_widget(BoxLayoutGame)
        ###     
        BoxLayoutInfo=BoxLayout(orientation='horizontal')
        ImageInfo=Image(source='command.jpg')
        BoxLayoutInfo.add_widget(ImageInfo)
        LabelInfo=Label(text='We are command from \nN.Novgorod,Russia.\nWe are Max and Anna.\nWe want to help \ndeaf-mute people,\nso we created\n this application.',font_size='25dp',font_name='RAVIE.ttf',color=(113.0/255,17.0/255,150.0/255,1))
        BoxLayoutInfo.add_widget(LabelInfo)
        info=AccordionItem(title='About us')
        info.add_widget(BoxLayoutInfo)

        ButtonExit=Button(text='Exit')
        ButtonExit.bind(on_press=go_exit)
        ButtonExit.size_hint=(1,.1)

        #rexit=AccordionItem(title='Exit')
        #rexit.add_widget(ButtonExit)
        
        root.add_widget(children)
        root.add_widget(game)
        root.add_widget(info)
        #root.add_widget(rexit)
        root.current=children
        Menu.add_widget(root)
        Menu.add_widget(ButtonExit)

        return Menu
    
################################
def _on_touch_down(x,y):
        global sm
        sm.current=sm.next()

def loading():
        Window.clearcolor=(1,140.0/255,15.0/255,0)
        #Window.fullscreen=True
        f=FloatLayout()
        title = Label(
            text='Trages',
            markup=True,
            bold=True,
            color=(79.0/255,15.0/255,204.0/255,0),
            font_name='RAVIE.ttf',
            halign='center',
            font_size='100dp',
            padding_y=2*Window.height/6
        )       
        img=Image(source='hand.png',keep_data=True)
        img.y=Window.height/6
        f.bind(on_touch_down=_on_touch_down)
        f.add_widget(title)
        f.add_widget(img)
        return f

class TestApp(App):
    def build(self):
        self.title='Trages'
        #Window.position=(0,0)
        # Create the screen manager
        global sm 
        sm = ScreenManager()
    
        sc1 = Screen(name='loading')
        f=loading()
        sc1.add_widget(f)

        sc2 = Screen(name='mainmenu')
        menu=main_menu()
        sc2.add_widget(menu)
        
        sm.add_widget(sc1)
        sm.add_widget(sc2)
        return sm
   
if __name__ == '__main__':
    # Create the screen manager

    TestApp().run()
