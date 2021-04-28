from django.http import HttpResponse
from django.shortcuts import render
from . import text_pred

def homepage(request):
    # return render(request,'homea.html')
    return render(request,'homepage.html')

# def result(request):
#     djtext=request.POST.get('text','default')
#     ans=text_pred.Bgen(djtext)
#     pred=ans.split()
#     print(pred[0])
#     Prediction={'data1': pred[0],'data2': pred[1],'data3': pred[2],'data4': pred[3],'data5': pred[4],'ans':djtext}
#     return render(request, 'result.html',Prediction )




def result(request):
    djtext=request.POST.get('text','default')
    Politics=request.POST.get('Politics','off')
    Business=request.POST.get('Business','off')
    Tech=request.POST.get('Tech','off')
    Sports=request.POST.get('Sports','off')
    Entertainment=request.POST.get('Entertainment','off')

    if(Business=="on"):
        
        ans=text_pred.Bgen(djtext)
        pred=ans.split()
        print(pred[0])
        Prediction={'data1': pred[0],'data2': pred[1],'data3': pred[2],'data4': pred[3],'data5': pred[4],'ans':djtext}
        print("Business run")

        
    elif Tech=="on":
        ans=text_pred.Tgen(djtext)
        pred=ans.split()
        print(pred[0])
        Prediction={'data1': pred[0],'data2': pred[1],'data3': pred[2],'data4': pred[3],'data5': pred[4],'ans':djtext}
        print("tech run")

    elif Sports=="on":
        ans=text_pred.Sgen(djtext)
        pred=ans.split()
        print(pred[0])
        Prediction={'data1': pred[0],'data2': pred[1],'data3': pred[2],'data4': pred[3],'data5': pred[4],'ans':djtext}
        print("Sports run")
    elif Entertainment=="on":
        ans=text_pred.Egen(djtext)
        pred=ans.split()
        print(pred[0])
        Prediction={'data1': pred[0],'data2': pred[1],'data3': pred[2],'data4': pred[3],'data5': pred[4],'ans':djtext}
        print("Entertainment run")
    elif Politics=="on":
        ans=text_pred.Pgen(djtext)
        pred=ans.split()
        print(pred[0])
        Prediction={'data1': pred[0],'data2': pred[1],'data3': pred[2],'data4': pred[3],'data5': pred[4],'ans':djtext}
        print("Politics run")
           
    return render(request, 'result.html',Prediction )

