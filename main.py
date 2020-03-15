import model



m = model.SIR(eons=4000, Susceptible = 3200, Infected= 200, rateIR= 0.03, rateSI= 0.01)
m.run()
m.plot()
