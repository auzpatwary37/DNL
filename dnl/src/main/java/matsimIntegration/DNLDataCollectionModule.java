package matsimIntegration;

import org.matsim.core.controler.AbstractModule;

import com.google.inject.Scope;
import com.google.inject.Singleton;
import com.google.inject.name.Names;

import linktolinkBPR.LinkToLinks;


public class DNLDataCollectionModule extends AbstractModule{
	
	private LinkToLinks l2ls;
	private String fileLoc;
	private String keyPrefix;
	private String keyFileLoc;
	private boolean instantenious;
	private String routeFileLoc;
	
	public DNLDataCollectionModule(LinkToLinks l2ls,String fileLoc,String keyPrefix,String keyFileloc,String routeDemandFileLoc, boolean instantenious) {
		this.l2ls=l2ls;
		this.fileLoc=fileLoc;
		this.keyPrefix=keyPrefix;
		this.keyFileLoc=keyFileloc;
		this.instantenious=instantenious;
		this.routeFileLoc = routeDemandFileLoc;
	}
	
	public void install() {
		bind(LinkToLinks.class).toInstance(this.l2ls);
		bind(LinkToLinkTTRecorder.class).toInstance(new LinkToLinkTTRecorder(this.l2ls));
		bind(String.class).annotatedWith(Names.named("fileLoc")).toInstance(this.fileLoc);
		bind(String.class).annotatedWith(Names.named("keyPrefix")).toInstance(this.keyPrefix);
		bind(String.class).annotatedWith(Names.named("keyFileloc")).toInstance(this.keyFileLoc);
		bind(String.class).annotatedWith(Names.named("routeDemandFileloc")).toInstance(this.routeFileLoc);
		bind(boolean.class).annotatedWith(Names.named("instantenious")).toInstance(this.instantenious);
		this.addEventHandlerBinding().to(LinkToLinkTTRecorder.class);
		this.addControlerListenerBinding().to(DNLDataCollectionControlerListener.class).in(Singleton.class);
		
	}

}
