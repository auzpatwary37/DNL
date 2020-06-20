package training;

import org.matsim.api.core.v01.population.Population;
import org.matsim.api.core.v01.population.PopulationWriter;
import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigWriter;
import org.matsim.vehicles.VehicleWriterV1;
import org.matsim.vehicles.Vehicles;

public class PopulationGenerationOutput{
	final Population population;
	final Vehicles vehicles;
	final Config config;
	
	public PopulationGenerationOutput(Population population, Vehicles vehicles, Config config) {
		this.population = population;
		this.vehicles = vehicles;
		this.config = config;
	}
	
	public void writeFiles(String populationLoc,String vehiclesLoc,String configLoc) {
		new ConfigWriter(config).write(configLoc);
		new PopulationWriter(population).write(populationLoc);
		new VehicleWriterV1(vehicles).writeFile(vehiclesLoc);
	}
}
